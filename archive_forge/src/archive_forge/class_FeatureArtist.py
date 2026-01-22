import warnings
import weakref
import matplotlib.artist
import matplotlib.collections
import matplotlib.path as mpath
import numpy as np
import cartopy.feature as cfeature
from cartopy.mpl import _MPL_38
import cartopy.mpl.patch as cpatch
class FeatureArtist(matplotlib.collections.Collection):
    """
    A subclass of :class:`~matplotlib.collections.Collection` capable of
    drawing a :class:`cartopy.feature.Feature`.

    """
    _geom_key_to_geometry_cache = weakref.WeakValueDictionary()
    '\n    A mapping from _GeomKey to geometry to assist with the caching of\n    transformed Matplotlib paths.\n\n    '
    _geom_key_to_path_cache = weakref.WeakKeyDictionary()
    '\n    A nested mapping from geometry (converted to a _GeomKey) and target\n    projection to the resulting transformed Matplotlib paths::\n\n        {geom: {target_projection: list_of_paths}}\n\n    This provides a significant boost when producing multiple maps of the\n    same projection.\n\n    '

    def __init__(self, feature, **kwargs):
        """
        Parameters
        ----------
        feature
            An instance of :class:`cartopy.feature.Feature` to draw.
        styler
            A callable that given a geometry, returns matplotlib styling
            parameters.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments to be used when drawing the feature. These
            will override those shared with the feature.

        """
        super().__init__()
        self._styler = kwargs.pop('styler', None)
        self._kwargs = dict(kwargs)
        if 'color' in self._kwargs:
            color = self._kwargs.pop('color')
            self._kwargs['facecolor'] = self._kwargs['edgecolor'] = color
        self.set_paths([])
        self.set_zorder(1.5)
        self.set(**feature.kwargs)
        self.set(**self._kwargs)
        self._feature = feature

    def set_facecolor(self, c):
        """
        Set the facecolor(s) of the `.FeatureArtist`.  If set to 'never' then
        subsequent calls will have no effect.  Otherwise works the same as
        `matplotlib.collections.Collection.set_facecolor`.
        """
        if isinstance(c, str) and c == 'never':
            self._never_fc = True
            super().set_facecolor('none')
        elif getattr(self, '_never_fc', False) and (not isinstance(c, str) or c != 'none'):
            warnings.warn('facecolor will have no effect as it has been defined as "never".')
        else:
            super().set_facecolor(c)
    if not _MPL_38:

        def set_paths(self, paths):
            self._paths = paths

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer):
        """
        Draw the geometries of the feature that intersect with the extent of
        the :class:`cartopy.mpl.geoaxes.GeoAxes` instance to which this
        object has been added.

        """
        if not self.get_visible():
            return
        ax = self.axes
        feature_crs = self._feature.crs
        extent = None
        try:
            extent = ax.get_extent(feature_crs)
        except ValueError:
            warnings.warn('Unable to determine extent. Defaulting to global.')
        if isinstance(self._feature, cfeature.ShapelyFeature):
            geoms = self._feature.geometries()
        else:
            geoms = self._feature.intersecting_geometries(extent)
        stylised_paths = {}
        no_style = _freeze({})
        key = ax.projection
        for geom in geoms:
            geom_key = _GeomKey(geom)
            FeatureArtist._geom_key_to_geometry_cache.setdefault(geom_key, geom)
            mapping = FeatureArtist._geom_key_to_path_cache.setdefault(geom_key, {})
            geom_path = mapping.get(key)
            if geom_path is None:
                if ax.projection != feature_crs:
                    projected_geom = ax.projection.project_geometry(geom, feature_crs)
                else:
                    projected_geom = geom
                geom_paths = cpatch.geos_to_path(projected_geom)
                geom_path = mpath.Path.make_compound_path(*geom_paths)
                mapping[key] = geom_path
            if self._styler is None:
                stylised_paths.setdefault(no_style, []).append(geom_path)
            else:
                style = _freeze(self._styler(geom))
                stylised_paths.setdefault(style, []).append(geom_path)
        self.set_clip_path(ax.patch)
        for style, paths in stylised_paths.items():
            style = dict(style)
            orig_style = {k: getattr(self, f'get_{k}')() for k in style}
            self.set(paths=paths, **style)
            super().draw(renderer)
            self.set(paths=[], **orig_style)