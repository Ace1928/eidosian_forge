from ... import exc
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_db
from ...testing.provision import generate_driver_url
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@drop_db.for_db('mysql', 'mariadb')
def _mysql_drop_db(cfg, eng, ident):
    with eng.begin() as conn:
        conn.exec_driver_sql('DROP DATABASE %s_test_schema' % ident)
        conn.exec_driver_sql('DROP DATABASE %s_test_schema_2' % ident)
        conn.exec_driver_sql('DROP DATABASE %s' % ident)