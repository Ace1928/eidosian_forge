from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
class CollectionOperator:

    def shapeup(self, ob):
        if isinstance(ob, BaseGeometry):
            return ob
        else:
            try:
                return shape(ob)
            except (ValueError, AttributeError):
                return LineString(ob)

    def polygonize(self, lines):
        """Creates polygons from a source of lines

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.
        """
        source = getattr(lines, 'geoms', None) or lines
        try:
            source = iter(source)
        except TypeError:
            source = [source]
        finally:
            obs = [self.shapeup(line) for line in source]
        collection = shapely.polygonize(obs)
        return collection.geoms

    def polygonize_full(self, lines):
        """Creates polygons from a source of lines, returning the polygons
        and leftover geometries.

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.

        Returns a tuple of objects: (polygons, cut edges, dangles, invalid ring
        lines). Each are a geometry collection.

        Dangles are edges which have one or both ends which are not incident on
        another edge endpoint. Cut edges are connected at both ends but do not
        form part of polygon. Invalid ring lines form rings which are invalid
        (bowties, etc).
        """
        source = getattr(lines, 'geoms', None) or lines
        try:
            source = iter(source)
        except TypeError:
            source = [source]
        finally:
            obs = [self.shapeup(line) for line in source]
        return shapely.polygonize_full(obs)

    def linemerge(self, lines, directed=False):
        """Merges all connected lines from a source

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.  Returns a
        LineString or MultiLineString when lines are not contiguous.
        """
        source = None
        if getattr(lines, 'geom_type', None) == 'MultiLineString':
            source = lines
        elif hasattr(lines, 'geoms'):
            source = MultiLineString([ls.coords for ls in lines.geoms])
        elif hasattr(lines, '__iter__'):
            try:
                source = MultiLineString([ls.coords for ls in lines])
            except AttributeError:
                source = MultiLineString(lines)
        if source is None:
            raise ValueError(f'Cannot linemerge {lines}')
        return shapely.line_merge(source, directed=directed)

    def cascaded_union(self, geoms):
        """Returns the union of a sequence of geometries

        .. deprecated:: 1.8
            This function was superseded by :meth:`unary_union`.
        """
        warn("The 'cascaded_union()' function is deprecated. Use 'unary_union()' instead.", ShapelyDeprecationWarning, stacklevel=2)
        return shapely.union_all(geoms, axis=None)

    def unary_union(self, geoms):
        """Returns the union of a sequence of geometries

        Usually used to convert a collection into the smallest set of polygons
        that cover the same area.
        """
        return shapely.union_all(geoms, axis=None)