import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Union, cast
from twisted.internet import defer
from twisted.internet.defer import Deferred
from scrapy import Request, Spider, signals
from scrapy.exceptions import NotConfigured, NotSupported
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.python import without_none_values
def _load_handler(self, scheme: str, skip_lazy: bool=False) -> Any:
    path = self._schemes[scheme]
    try:
        dhcls = load_object(path)
        if skip_lazy and getattr(dhcls, 'lazy', True):
            return None
        dh = create_instance(objcls=dhcls, settings=self._crawler.settings, crawler=self._crawler)
    except NotConfigured as ex:
        self._notconfigured[scheme] = str(ex)
        return None
    except Exception as ex:
        logger.error('Loading "%(clspath)s" for scheme "%(scheme)s"', {'clspath': path, 'scheme': scheme}, exc_info=True, extra={'crawler': self._crawler})
        self._notconfigured[scheme] = str(ex)
        return None
    else:
        self._handlers[scheme] = dh
        return dh