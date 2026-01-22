from twisted.internet import threads
from twisted.python import log, reflect
def _runWithConnection(self, func, *args, **kw):
    conn = self.connectionFactory(self)
    try:
        result = func(conn, *args, **kw)
        conn.commit()
        return result
    except BaseException:
        try:
            conn.rollback()
        except BaseException:
            log.err(None, 'Rollback failed')
        raise