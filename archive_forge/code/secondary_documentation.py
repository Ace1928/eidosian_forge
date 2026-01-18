from twisted.application import service
from twisted.internet import defer, task
from twisted.names import client, common, dns, resolve
from twisted.names.authority import FileAuthority
from twisted.python import failure, log
from twisted.python.compat import nativeString

        Attempt a zone transfer.

        @returns: A L{Deferred} that fires with L{None} when attempted zone
            transfer has completed.
        