import ipaddress
import logging
import re
from contextlib import suppress
from io import BytesIO
from time import time
from urllib.parse import urldefrag, urlunparse
from twisted.internet import defer, protocol, ssl
from twisted.internet.endpoints import TCP4ClientEndpoint
from twisted.internet.error import TimeoutError
from twisted.python.failure import Failure
from twisted.web.client import (
from twisted.web.http import PotentialDataLoss, _DataLoss
from twisted.web.http_headers import Headers as TxHeaders
from twisted.web.iweb import UNKNOWN_LENGTH, IBodyProducer
from zope.interface import implementer
from scrapy import signals
from scrapy.core.downloader.contextfactory import load_context_factory_from_settings
from scrapy.core.downloader.webclient import _parse
from scrapy.exceptions import StopDownload
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.python import to_bytes, to_unicode
def _cb_bodyready(self, txresponse, request):
    headers_received_result = self._crawler.signals.send_catch_log(signal=signals.headers_received, headers=self._headers_from_twisted_response(txresponse), body_length=txresponse.length, request=request, spider=self._crawler.spider)
    for handler, result in headers_received_result:
        if isinstance(result, Failure) and isinstance(result.value, StopDownload):
            logger.debug('Download stopped for %(request)s from signal handler %(handler)s', {'request': request, 'handler': handler.__qualname__})
            txresponse._transport.stopProducing()
            txresponse._transport.loseConnection()
            return {'txresponse': txresponse, 'body': b'', 'flags': ['download_stopped'], 'certificate': None, 'ip_address': None, 'failure': result if result.value.fail else None}
    if txresponse.length == 0:
        return {'txresponse': txresponse, 'body': b'', 'flags': None, 'certificate': None, 'ip_address': None}
    maxsize = request.meta.get('download_maxsize', self._maxsize)
    warnsize = request.meta.get('download_warnsize', self._warnsize)
    expected_size = txresponse.length if txresponse.length != UNKNOWN_LENGTH else -1
    fail_on_dataloss = request.meta.get('download_fail_on_dataloss', self._fail_on_dataloss)
    if maxsize and expected_size > maxsize:
        warning_msg = 'Cancelling download of %(url)s: expected response size (%(size)s) larger than download max size (%(maxsize)s).'
        warning_args = {'url': request.url, 'size': expected_size, 'maxsize': maxsize}
        logger.warning(warning_msg, warning_args)
        txresponse._transport.loseConnection()
        raise defer.CancelledError(warning_msg % warning_args)
    if warnsize and expected_size > warnsize:
        logger.warning('Expected response size (%(size)s) larger than download warn size (%(warnsize)s) in request %(request)s.', {'size': expected_size, 'warnsize': warnsize, 'request': request})

    def _cancel(_):
        txresponse._transport._producer.abortConnection()
    d = defer.Deferred(_cancel)
    txresponse.deliverBody(_ResponseReader(finished=d, txresponse=txresponse, request=request, maxsize=maxsize, warnsize=warnsize, fail_on_dataloss=fail_on_dataloss, crawler=self._crawler))
    self._txresponse = txresponse
    return d