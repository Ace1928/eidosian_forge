from django.contrib.gis.gdal import OGRGeomType
from django.db.backends.postgresql.introspection import DatabaseIntrospection
class PostGISIntrospection(DatabaseIntrospection):
    postgis_oid_lookup = {}
    ignored_tables = DatabaseIntrospection.ignored_tables + ['geography_columns', 'geometry_columns', 'raster_columns', 'spatial_ref_sys', 'raster_overviews']

    def get_field_type(self, data_type, description):
        if not self.postgis_oid_lookup:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT oid, typname FROM pg_type WHERE typname IN ('geometry', 'geography')")
                self.postgis_oid_lookup = dict(cursor.fetchall())
            self.data_types_reverse.update(((oid, 'GeometryField') for oid in self.postgis_oid_lookup))
        return super().get_field_type(data_type, description)

    def get_geometry_type(self, table_name, description):
        """
        The geometry type OID used by PostGIS does not indicate the particular
        type of field that a geometry column is (e.g., whether it's a
        PointField or a PolygonField).  Thus, this routine queries the PostGIS
        metadata tables to determine the geometry type.
        """
        with self.connection.cursor() as cursor:
            cursor.execute('\n                SELECT t.coord_dimension, t.srid, t.type FROM (\n                    SELECT * FROM geometry_columns\n                    UNION ALL\n                    SELECT * FROM geography_columns\n                ) AS t WHERE t.f_table_name = %s AND t.f_geometry_column = %s\n            ', (table_name, description.name))
            row = cursor.fetchone()
            if not row:
                raise Exception('Could not find a geometry or geography column for "%s"."%s"' % (table_name, description.name))
            dim, srid, field_type = row
            field_type = OGRGeomType(field_type).django
            field_params = {}
            if self.postgis_oid_lookup.get(description.type_code) == 'geography':
                field_params['geography'] = True
            if srid != 4326:
                field_params['srid'] = srid
            if dim != 2:
                field_params['dim'] = dim
        return (field_type, field_params)