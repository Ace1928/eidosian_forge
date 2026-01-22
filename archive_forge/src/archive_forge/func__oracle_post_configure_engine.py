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
@post_configure_engine.for_db('oracle')
def _oracle_post_configure_engine(url, engine, follower_ident):
    from sqlalchemy import event

    @event.listens_for(engine, 'checkout')
    def checkout(dbapi_con, con_record, con_proxy):
        _all_conns.add(dbapi_con)

    @event.listens_for(engine, 'checkin')
    def checkin(dbapi_connection, connection_record):
        if 'cx_oracle_xid' in connection_record.info:
            connection_record.invalidate()