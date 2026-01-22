from .psycopg2 import PGDialect_psycopg2
from ... import util
class PGDialect_psycopg2cffi(PGDialect_psycopg2):
    driver = 'psycopg2cffi'
    supports_unicode_statements = True
    supports_statement_cache = True
    FEATURE_VERSION_MAP = dict(native_json=(2, 4, 4), native_jsonb=(2, 7, 1), sane_multi_rowcount=(2, 4, 4), array_oid=(2, 4, 4), hstore_adapter=(2, 4, 4))

    @classmethod
    def import_dbapi(cls):
        return __import__('psycopg2cffi')

    @util.memoized_property
    def _psycopg2_extensions(cls):
        root = __import__('psycopg2cffi', fromlist=['extensions'])
        return root.extensions

    @util.memoized_property
    def _psycopg2_extras(cls):
        root = __import__('psycopg2cffi', fromlist=['extras'])
        return root.extras