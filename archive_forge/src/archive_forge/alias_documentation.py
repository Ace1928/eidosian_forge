import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log

        Map each of the aliases in the group to its ultimate destination.

        @type aliasmap: L{dict} mapping L{bytes} to L{AliasBase}
        @param aliasmap: A mapping of username to alias or group of aliases.

        @type memo: L{None} or L{dict} of L{AliasBase}
        @param memo: A record of the aliases already considered in the
            resolution process.  If provided, C{memo} is modified to include
            this alias.

        @rtype: L{MultiWrapper}
        @return: A message receiver which passes the message on to message
            receivers for the ultimate destination of each alias in the group.
        