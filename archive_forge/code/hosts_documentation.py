from twisted.internet import defer
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.names import common, dns
from twisted.python import failure
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath

        Read any IPv6 addresses from C{self.file} and return them as
        L{Record_AAAA} instances.
        