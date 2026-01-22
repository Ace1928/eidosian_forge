from functools import lru_cache
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.base import DatabaseWrapper as PsycopgDatabaseWrapper
from django.db.backends.postgresql.features import (
from django.db.backends.postgresql.introspection import (
from django.db.backends.postgresql.operations import (
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from .adapter import PostGISAdapter
from .features import DatabaseFeatures
from .introspection import PostGISIntrospection
from .operations import PostGISOperations
from .schema import PostGISSchemaEditor
class DatabaseWrapper(PsycopgDatabaseWrapper):
    SchemaEditorClass = PostGISSchemaEditor
    features_class = DatabaseFeatures
    ops_class = PostGISOperations
    introspection_class = PostGISIntrospection
    _type_infos = {'geometry': {}, 'geography': {}, 'raster': {}}

    def __init__(self, *args, **kwargs):
        if kwargs.get('alias', '') == NO_DB_ALIAS:
            self.features_class = PsycopgDatabaseFeatures
            self.ops_class = PsycopgDatabaseOperations
            self.introspection_class = PsycopgDatabaseIntrospection
        super().__init__(*args, **kwargs)

    def prepare_database(self):
        super().prepare_database()
        with self.cursor() as cursor:
            cursor.execute('SELECT 1 FROM pg_extension WHERE extname = %s', ['postgis'])
            if bool(cursor.fetchone()):
                return
            cursor.execute('CREATE EXTENSION IF NOT EXISTS postgis')
            if is_psycopg3:
                self.register_geometry_adapters(self.connection, True)

    def get_new_connection(self, conn_params):
        connection = super().get_new_connection(conn_params)
        if is_psycopg3:
            self.register_geometry_adapters(connection)
        return connection
    if is_psycopg3:

        def _register_type(self, pg_connection, typename):
            registry = self._type_infos[typename]
            try:
                info = registry[self.alias]
            except KeyError:
                info = TypeInfo.fetch(pg_connection, typename)
                registry[self.alias] = info
            if info:
                info.register(pg_connection)
                pg_connection.adapters.register_loader(info.oid, TextLoader)
                pg_connection.adapters.register_loader(info.oid, TextBinaryLoader)
            return info.oid if info else None

        def register_geometry_adapters(self, pg_connection, clear_caches=False):
            if clear_caches:
                for typename in self._type_infos:
                    self._type_infos[typename].pop(self.alias, None)
            geo_oid = self._register_type(pg_connection, 'geometry')
            geog_oid = self._register_type(pg_connection, 'geography')
            raster_oid = self._register_type(pg_connection, 'raster')
            PostGISTextDumper, PostGISBinaryDumper = postgis_adapters(geo_oid, geog_oid, raster_oid)
            pg_connection.adapters.register_dumper(PostGISAdapter, PostGISTextDumper)
            pg_connection.adapters.register_dumper(PostGISAdapter, PostGISBinaryDumper)