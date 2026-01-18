from typing import Callable, Dict, Iterable, List, Tuple, Type, Union
from zope.interface import Interface, providedBy
from twisted.cred import error
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import ICredentials
from twisted.internet import defer
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.python import failure, reflect

        @param credentials: an implementor of
            L{twisted.cred.credentials.ICredentials}

        @param mind: an object which implements a client-side interface for
            your particular realm.  In many cases, this may be None, so if the
            word 'mind' confuses you, just ignore it.

        @param interfaces: list of interfaces for the perspective that the mind
            wishes to attach to. Usually, this will be only one interface, for
            example IMailAccount. For highly dynamic protocols, however, this
            may be a list like (IMailAccount, IUserChooser, IServiceInfo).  To
            expand: if we are speaking to the system over IMAP, any information
            that will be relayed to the user MUST be returned as an
            IMailAccount implementor; IMAP clients would not be able to
            understand anything else. Any information about unusual status
            would have to be relayed as a single mail message in an
            otherwise-empty mailbox. However, in a web-based mail system, or a
            PB-based client, the ``mind'' object inside the web server
            (implemented with a dynamic page-viewing mechanism such as a
            Twisted Web Resource) or on the user's client program may be
            intelligent enough to respond to several ``server''-side
            interfaces.

        @return: A deferred which will fire a tuple of (interface,
            avatarAspect, logout).  The interface will be one of the interfaces
            passed in the 'interfaces' argument.  The 'avatarAspect' will
            implement that interface. The 'logout' object is a callable which
            will detach the mind from the avatar. It must be called when the
            user has conceptually disconnected from the service. Although in
            some cases this will not be in connectionLost (such as in a
            web-based session), it will always be at the end of a user's
            interactive session.
        