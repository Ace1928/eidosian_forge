from twisted.application import service
from twisted.internet import defer, task
from twisted.names import client, common, dns, resolve
from twisted.names.authority import FileAuthority
from twisted.python import failure, log
from twisted.python.compat import nativeString
def _ebTransferred(self, failure):
    self.transferred = False
    log.msg('Transferring %s from %s failed after zone transfer' % (self.domain, self.primary))
    log.err(failure)