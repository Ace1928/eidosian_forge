import logging
import sys
import param
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from holoviews.core.data import MultiInterface
from holoviews.core.util import cartesian_product, get_param_values
from holoviews.operation import Operation
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from ..data import GeoPandasInterface
from ..element import (Image, Shape, Polygons, Path, Points, Contours,
from ..util import (
class project_path(_project_operation):
    """
    Projects Polygons and Path Elements from their source coordinate
    reference system to the supplied projection.
    """
    supported_types = [Polygons, Path, Contours, EdgePaths]

    def _process_element(self, element):
        if not bool(element):
            return element.clone(crs=self.p.projection)
        crs = element.crs
        proj = self.p.projection
        if isinstance(crs, ccrs.PlateCarree) and (not isinstance(proj, ccrs.PlateCarree)) and (crs.proj4_params['lon_0'] != 0):
            element = self.instance(projection=ccrs.PlateCarree())(element)
        if isinstance(proj, ccrs.CRS) and (not isinstance(proj, ccrs.Projection)):
            raise ValueError('invalid transform: Spherical contouring is not supported -  consider using PlateCarree/RotatedPole.')
        if isinstance(element, Polygons):
            geoms = polygons_to_geom_dicts(element, skip_invalid=False)
        else:
            geoms = path_to_geom_dicts(element, skip_invalid=False)
        projected = []
        for path in geoms:
            geom = path['geometry']
            if isinstance(geom, Polygon) and geom.area < 1e-15:
                continue
            elif isinstance(geom, MultiPolygon):
                polys = [g for g in geom.geoms if g.area > 1e-15]
                if not polys:
                    continue
                geom = MultiPolygon(polys)
            elif not geom or isinstance(geom, GeometryCollection):
                continue
            proj_geom = proj.project_geometry(geom, element.crs)
            logger = logging.getLogger()
            try:
                prev = logger.level
                logger.setLevel(logging.ERROR)
                if not proj_geom.is_valid:
                    proj_geom = proj.project_geometry(geom.buffer(0), element.crs)
            except Exception:
                continue
            finally:
                logger.setLevel(prev)
            if proj_geom.geom_type in ['GeometryCollection', 'MultiPolygon'] and len(proj_geom.geoms) == 0:
                continue
            data = dict(path, geometry=proj_geom)
            if 'holes' in data:
                data.pop('holes')
            projected.append(data)
        if len(geoms) and len(projected) == 0:
            element_name = type(element).__name__
            crs_name = type(element.crs).__name__
            proj_name = type(self.p.projection).__name__
            self.param.warning(f'While projecting a {element_name} element from a {crs_name} coordinate reference system (crs) to a {proj_name} projection none of the projected paths were contained within the bounds specified by the projection. Ensure you have specified the correct coordinate system for your data.')
        if element.interface is GeoPandasInterface:
            import geopandas as gpd
            projected = gpd.GeoDataFrame(projected, columns=element.data.columns)
        elif element.interface is MultiInterface:
            x, y = element.kdims
            item = element.data[0] if element.data else None
            if item is None or (isinstance(item, dict) and 'geometry' in item):
                return element.clone(projected, crs=self.p.projection)
            projected = [geom_dict_to_array_dict(p, [x.name, y.name]) for p in projected]
            if any(('holes' in p for p in projected)):
                pass
            elif pd and isinstance(item, pd.DataFrame):
                projected = [pd.DataFrame(p, columns=item.columns) for p in projected]
            elif isinstance(item, np.ndarray):
                projected = [np.column_stack([p[d.name] for d in element.dimensions()]) for p in projected]
        return element.clone(projected, crs=self.p.projection)