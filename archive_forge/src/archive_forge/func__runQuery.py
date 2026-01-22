from twisted.internet import threads
from twisted.python import log, reflect
def _runQuery(self, trans, *args, **kw):
    trans.execute(*args, **kw)
    return trans.fetchall()