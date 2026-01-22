import os
import pickle
from twisted.internet.address import UNIXAddress
from twisted.mail import smtp
from twisted.python import log
class ESMTPRelayer(RelayerMixin, smtp.ESMTPClient):
    """
    A base class for ESMTP relayers.
    """

    def __init__(self, messagePaths, *args, **kw):
        """
        @type messagePaths: L{list} of L{bytes}
        @param messagePaths: The base filename for each message to be relayed.

        @type args: 3-L{tuple} of (0) L{bytes}, (1) L{None} or
            L{ClientContextFactory
            <twisted.internet.ssl.ClientContextFactory>},
            (2) L{bytes} or 4-L{tuple} of (0) L{bytes}, (1) L{None}
            or L{ClientContextFactory
            <twisted.internet.ssl.ClientContextFactory>}, (2) L{bytes},
            (3) L{int}
        @param args: Positional arguments for L{ESMTPClient.__init__}

        @type kw: L{dict}
        @param kw: Keyword arguments for L{ESMTPClient.__init__}
        """
        smtp.ESMTPClient.__init__(self, *args, **kw)
        self.loadMessages(messagePaths)