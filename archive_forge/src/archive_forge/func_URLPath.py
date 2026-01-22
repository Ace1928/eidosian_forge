from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
def URLPath():
    """
        @return: A L{URLPath<twisted.python.urlpath.URLPath>} instance
            which identifies the URL for which this request is.
        """