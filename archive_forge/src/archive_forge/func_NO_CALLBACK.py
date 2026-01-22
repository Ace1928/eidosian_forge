import inspect
from typing import Callable, List, Optional, Tuple, Type, TypeVar, Union
from w3lib.url import safe_url_string
import scrapy
from scrapy.http.common import obsolete_setter
from scrapy.http.headers import Headers
from scrapy.utils.curl import curl_to_request_kwargs
from scrapy.utils.python import to_bytes
from scrapy.utils.trackref import object_ref
from scrapy.utils.url import escape_ajax
def NO_CALLBACK(*args, **kwargs):
    """When assigned to the ``callback`` parameter of
    :class:`~scrapy.http.Request`, it indicates that the request is not meant
    to have a spider callback at all.

    For example:

    .. code-block:: python

       Request("https://example.com", callback=NO_CALLBACK)

    This value should be used by :ref:`components <topics-components>` that
    create and handle their own requests, e.g. through
    :meth:`scrapy.core.engine.ExecutionEngine.download`, so that downloader
    middlewares handling such requests can treat them differently from requests
    intended for the :meth:`~scrapy.Spider.parse` callback.
    """
    raise RuntimeError('The NO_CALLBACK callback has been called. This is a special callback value intended for requests whose callback is never meant to be called.')