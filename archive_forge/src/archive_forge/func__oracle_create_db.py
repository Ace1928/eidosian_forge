from ... import create_engine
from ... import exc
from ... import inspect
from ...engine import url as sa_url
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import follower_url_from_main
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import run_reap_dbs
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import stop_test_class_outside_fixtures
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import update_db_opts
@create_db.for_db('oracle')
def _oracle_create_db(cfg, eng, ident):
    with eng.begin() as conn:
        conn.exec_driver_sql('create user %s identified by xe' % ident)
        conn.exec_driver_sql('create user %s_ts1 identified by xe' % ident)
        conn.exec_driver_sql('create user %s_ts2 identified by xe' % ident)
        conn.exec_driver_sql('grant dba to %s' % (ident,))
        conn.exec_driver_sql('grant unlimited tablespace to %s' % ident)
        conn.exec_driver_sql('grant unlimited tablespace to %s_ts1' % ident)
        conn.exec_driver_sql('grant unlimited tablespace to %s_ts2' % ident)
        conn.exec_driver_sql('grant create table to %s' % ident)
        conn.exec_driver_sql('grant create table to %s_ts1' % ident)
        conn.exec_driver_sql('grant create table to %s_ts2' % ident)