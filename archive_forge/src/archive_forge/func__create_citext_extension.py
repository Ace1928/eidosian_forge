import time
from ... import exc
from ... import inspect
from ... import text
from ...testing import warn_test_suite
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import prepare_for_drop_tables
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@post_configure_engine.for_db('postgresql')
def _create_citext_extension(url, engine, follower_ident):
    with engine.connect() as conn:
        for extension, min_version in _extensions:
            if conn.dialect.server_version_info >= min_version:
                conn.execute(text(f'CREATE EXTENSION IF NOT EXISTS {extension}'))
                conn.commit()