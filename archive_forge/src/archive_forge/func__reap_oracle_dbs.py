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
@run_reap_dbs.for_db('oracle')
def _reap_oracle_dbs(url, idents):
    log.info('db reaper connecting to %r', url)
    eng = create_engine(url)
    with eng.begin() as conn:
        log.info('identifiers in file: %s', ', '.join(idents))
        to_reap = conn.exec_driver_sql("select u.username from all_users u where username like 'TEST_%' and not exists (select username from v$session where username=u.username)")
        all_names = {username.lower() for username, in to_reap}
        to_drop = set()
        for name in all_names:
            if name.endswith('_ts1') or name.endswith('_ts2'):
                continue
            elif name in idents:
                to_drop.add(name)
                if '%s_ts1' % name in all_names:
                    to_drop.add('%s_ts1' % name)
                if '%s_ts2' % name in all_names:
                    to_drop.add('%s_ts2' % name)
        dropped = total = 0
        for total, username in enumerate(to_drop, 1):
            if _ora_drop_ignore(conn, username):
                dropped += 1
        log.info('Dropped %d out of %d stale databases detected', dropped, total)