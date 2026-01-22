import warnings
from typing import TYPE_CHECKING, Any, List, Optional
from OpenSSL import SSL
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.ssl import (
from twisted.web.client import BrowserLikePolicyForHTTPS
from twisted.web.iweb import IPolicyForHTTPS
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from scrapy.core.downloader.tls import (
from scrapy.settings import BaseSettings
from scrapy.utils.misc import create_instance, load_object
@implementer(IPolicyForHTTPS)
class BrowserLikeContextFactory(ScrapyClientContextFactory):
    """
    Twisted-recommended context factory for web clients.

    Quoting the documentation of the :class:`~twisted.web.client.Agent` class:

        The default is to use a
        :class:`~twisted.web.client.BrowserLikePolicyForHTTPS`, so unless you
        have special requirements you can leave this as-is.

    :meth:`creatorForNetloc` is the same as
    :class:`~twisted.web.client.BrowserLikePolicyForHTTPS` except this context
    factory allows setting the TLS/SSL method to use.

    The default OpenSSL method is ``TLS_METHOD`` (also called
    ``SSLv23_METHOD``) which allows TLS protocol negotiation.
    """

    def creatorForNetloc(self, hostname: bytes, port: int) -> 'ClientTLSOptions':
        return optionsForClientTLS(hostname=hostname.decode('ascii'), trustRoot=platformTrust(), extraCertificateOptions={'method': self._ssl_method})