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
@create_db.for_db('postgresql')
def _pg_create_db(cfg, eng, ident):
    template_db = cfg.options.postgresql_templatedb
    with eng.execution_options(isolation_level='AUTOCOMMIT').begin() as conn:
        if not template_db:
            template_db = conn.exec_driver_sql('select current_database()').scalar()
        attempt = 0
        while True:
            try:
                conn.exec_driver_sql('CREATE DATABASE %s TEMPLATE %s' % (ident, template_db))
            except exc.OperationalError as err:
                attempt += 1
                if attempt >= 3:
                    raise
                if 'accessed by other users' in str(err):
                    log.info('Waiting to create %s, URI %r, template DB %s is in use sleeping for .5', ident, eng.url, template_db)
                    time.sleep(0.5)
            except:
                raise
            else:
                break