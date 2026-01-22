from ... import exc
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_db
from ...testing.provision import generate_driver_url
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@create_db.for_db('mysql', 'mariadb')
def _mysql_create_db(cfg, eng, ident):
    with eng.begin() as conn:
        try:
            _mysql_drop_db(cfg, conn, ident)
        except Exception:
            pass
    with eng.begin() as conn:
        conn.exec_driver_sql('CREATE DATABASE %s CHARACTER SET utf8mb4' % ident)
        conn.exec_driver_sql('CREATE DATABASE %s_test_schema CHARACTER SET utf8mb4' % ident)
        conn.exec_driver_sql('CREATE DATABASE %s_test_schema_2 CHARACTER SET utf8mb4' % ident)