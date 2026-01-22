import re
from django.conf import settings
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.db.models import GeometryField, RasterField
from django.contrib.gis.gdal import GDALRaster
from django.contrib.gis.geos.geometry import GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.core.exceptions import ImproperlyConfigured
from django.db import NotSupportedError, ProgrammingError
from django.db.backends.postgresql.operations import DatabaseOperations
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.db.models import Func, Value
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
from .adapter import PostGISAdapter
from .models import PostGISGeometryColumns, PostGISSpatialRefSys
from .pgraster import from_pgraster
class PostGISOperations(BaseSpatialOperations, DatabaseOperations):
    name = 'postgis'
    postgis = True
    geom_func_prefix = 'ST_'
    Adapter = PostGISAdapter
    collect = geom_func_prefix + 'Collect'
    extent = geom_func_prefix + 'Extent'
    extent3d = geom_func_prefix + '3DExtent'
    length3d = geom_func_prefix + '3DLength'
    makeline = geom_func_prefix + 'MakeLine'
    perimeter3d = geom_func_prefix + '3DPerimeter'
    unionagg = geom_func_prefix + 'Union'
    gis_operators = {'bbcontains': PostGISOperator(op='~', raster=True), 'bboverlaps': PostGISOperator(op='&&', geography=True, raster=True), 'contained': PostGISOperator(op='@', raster=True), 'overlaps_left': PostGISOperator(op='&<', raster=BILATERAL), 'overlaps_right': PostGISOperator(op='&>', raster=BILATERAL), 'overlaps_below': PostGISOperator(op='&<|'), 'overlaps_above': PostGISOperator(op='|&>'), 'left': PostGISOperator(op='<<'), 'right': PostGISOperator(op='>>'), 'strictly_below': PostGISOperator(op='<<|'), 'strictly_above': PostGISOperator(op='|>>'), 'same_as': PostGISOperator(op='~=', raster=BILATERAL), 'exact': PostGISOperator(op='~=', raster=BILATERAL), 'contains': PostGISOperator(func='ST_Contains', raster=BILATERAL), 'contains_properly': PostGISOperator(func='ST_ContainsProperly', raster=BILATERAL), 'coveredby': PostGISOperator(func='ST_CoveredBy', geography=True, raster=BILATERAL), 'covers': PostGISOperator(func='ST_Covers', geography=True, raster=BILATERAL), 'crosses': PostGISOperator(func='ST_Crosses'), 'disjoint': PostGISOperator(func='ST_Disjoint', raster=BILATERAL), 'equals': PostGISOperator(func='ST_Equals'), 'intersects': PostGISOperator(func='ST_Intersects', geography=True, raster=BILATERAL), 'overlaps': PostGISOperator(func='ST_Overlaps', raster=BILATERAL), 'relate': PostGISOperator(func='ST_Relate'), 'touches': PostGISOperator(func='ST_Touches', raster=BILATERAL), 'within': PostGISOperator(func='ST_Within', raster=BILATERAL), 'dwithin': PostGISOperator(func='ST_DWithin', geography=True, raster=BILATERAL)}
    unsupported_functions = set()
    select = '%s' if is_psycopg3 else '%s::bytea'
    select_extent = None

    @cached_property
    def function_names(self):
        function_names = {'AsWKB': 'ST_AsBinary', 'AsWKT': 'ST_AsText', 'BoundingCircle': 'ST_MinimumBoundingCircle', 'FromWKB': 'ST_GeomFromWKB', 'FromWKT': 'ST_GeomFromText', 'NumPoints': 'ST_NPoints'}
        return function_names

    @cached_property
    def spatial_version(self):
        """Determine the version of the PostGIS library."""
        if hasattr(settings, 'POSTGIS_VERSION'):
            version = settings.POSTGIS_VERSION
        else:
            self._get_postgis_func('version')
            try:
                vtup = self.postgis_version_tuple()
            except ProgrammingError:
                raise ImproperlyConfigured('Cannot determine PostGIS version for database "%s" using command "SELECT postgis_lib_version()". GeoDjango requires at least PostGIS version 2.5. Was the database created from a spatial database template?' % self.connection.settings_dict['NAME'])
            version = vtup[1:]
        return version

    def convert_extent(self, box):
        """
        Return a 4-tuple extent for the `Extent` aggregate by converting
        the bounding box text returned by PostGIS (`box` argument), for
        example: "BOX(-90.0 30.0, -85.0 40.0)".
        """
        if box is None:
            return None
        ll, ur = box[4:-1].split(',')
        xmin, ymin = map(float, ll.split())
        xmax, ymax = map(float, ur.split())
        return (xmin, ymin, xmax, ymax)

    def convert_extent3d(self, box3d):
        """
        Return a 6-tuple extent for the `Extent3D` aggregate by converting
        the 3d bounding-box text returned by PostGIS (`box3d` argument), for
        example: "BOX3D(-90.0 30.0 1, -85.0 40.0 2)".
        """
        if box3d is None:
            return None
        ll, ur = box3d[6:-1].split(',')
        xmin, ymin, zmin = map(float, ll.split())
        xmax, ymax, zmax = map(float, ur.split())
        return (xmin, ymin, zmin, xmax, ymax, zmax)

    def geo_db_type(self, f):
        """
        Return the database field type for the given spatial field.
        """
        if f.geom_type == 'RASTER':
            return 'raster'
        if f.dim == 3:
            geom_type = f.geom_type + 'Z'
        else:
            geom_type = f.geom_type
        if f.geography:
            if f.srid != 4326:
                raise NotSupportedError('PostGIS only supports geography columns with an SRID of 4326.')
            return 'geography(%s,%d)' % (geom_type, f.srid)
        else:
            return 'geometry(%s,%d)' % (geom_type, f.srid)

    def get_distance(self, f, dist_val, lookup_type):
        """
        Retrieve the distance parameters for the given geometry field,
        distance lookup value, and the distance lookup type.

        This is the most complex implementation of the spatial backends due to
        what is supported on geodetic geometry columns vs. what's available on
        projected geometry columns.  In addition, it has to take into account
        the geography column type.
        """
        value = dist_val[0]
        geodetic = f.geodetic(self.connection)
        geography = f.geography
        if isinstance(value, Distance):
            if geography:
                dist_param = value.m
            elif geodetic:
                if lookup_type == 'dwithin':
                    raise ValueError('Only numeric values of degree units are allowed on geographic DWithin queries.')
                dist_param = value.m
            else:
                dist_param = getattr(value, Distance.unit_attname(f.units_name(self.connection)))
        else:
            dist_param = value
        return [dist_param]

    def get_geom_placeholder(self, f, value, compiler):
        """
        Provide a proper substitution value for Geometries or rasters that are
        not in the SRID of the field. Specifically, this routine will
        substitute in the ST_Transform() function call.
        """
        transform_func = self.spatial_function_name('Transform')
        if hasattr(value, 'as_sql'):
            if value.field.srid == f.srid:
                placeholder = '%s'
            else:
                placeholder = '%s(%%s, %s)' % (transform_func, f.srid)
            return placeholder
        if value is None:
            value_srid = None
        else:
            value_srid = value.srid
        if value_srid is None or value_srid == f.srid:
            placeholder = '%s'
        else:
            placeholder = '%s(%%s, %s)' % (transform_func, f.srid)
        return placeholder

    def _get_postgis_func(self, func):
        """
        Helper routine for calling PostGIS functions and returning their result.
        """
        with self.connection.temporary_connection() as cursor:
            cursor.execute('SELECT %s()' % func)
            return cursor.fetchone()[0]

    def postgis_geos_version(self):
        """Return the version of the GEOS library used with PostGIS."""
        return self._get_postgis_func('postgis_geos_version')

    def postgis_lib_version(self):
        """Return the version number of the PostGIS library used with PostgreSQL."""
        return self._get_postgis_func('postgis_lib_version')

    def postgis_proj_version(self):
        """Return the version of the PROJ library used with PostGIS."""
        return self._get_postgis_func('postgis_proj_version')

    def postgis_version(self):
        """Return PostGIS version number and compile-time options."""
        return self._get_postgis_func('postgis_version')

    def postgis_full_version(self):
        """Return PostGIS version number and compile-time options."""
        return self._get_postgis_func('postgis_full_version')

    def postgis_version_tuple(self):
        """
        Return the PostGIS version as a tuple (version string, major,
        minor, subminor).
        """
        version = self.postgis_lib_version()
        return (version,) + get_version_tuple(version)

    def proj_version_tuple(self):
        """
        Return the version of PROJ used by PostGIS as a tuple of the
        major, minor, and subminor release numbers.
        """
        proj_regex = re.compile('(\\d+)\\.(\\d+)\\.(\\d+)')
        proj_ver_str = self.postgis_proj_version()
        m = proj_regex.search(proj_ver_str)
        if m:
            return tuple(map(int, m.groups()))
        else:
            raise Exception('Could not determine PROJ version from PostGIS.')

    def spatial_aggregate_name(self, agg_name):
        if agg_name == 'Extent3D':
            return self.extent3d
        else:
            return self.geom_func_prefix + agg_name

    def geometry_columns(self):
        return PostGISGeometryColumns

    def spatial_ref_sys(self):
        return PostGISSpatialRefSys

    def parse_raster(self, value):
        """Convert a PostGIS HEX String into a dict readable by GDALRaster."""
        return from_pgraster(value)

    def distance_expr_for_lookup(self, lhs, rhs, **kwargs):
        return super().distance_expr_for_lookup(self._normalize_distance_lookup_arg(lhs), self._normalize_distance_lookup_arg(rhs), **kwargs)

    @staticmethod
    def _normalize_distance_lookup_arg(arg):
        is_raster = arg.field.geom_type == 'RASTER' if hasattr(arg, 'field') else isinstance(arg, GDALRaster)
        return ST_Polygon(arg) if is_raster else arg

    def get_geometry_converter(self, expression):
        read = wkb_r().read
        geom_class = expression.output_field.geom_class

        def converter(value, expression, connection):
            if isinstance(value, str):
                value = value.encode('ascii')
            return None if value is None else GEOSGeometryBase(read(value), geom_class)
        return converter

    def get_area_att_for_field(self, field):
        return 'sq_m'