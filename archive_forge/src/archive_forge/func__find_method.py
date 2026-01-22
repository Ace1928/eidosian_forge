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
def _find_method(obj, func):
    """Helper function for Request.to_dict"""
    if obj and hasattr(func, '__func__'):
        members = inspect.getmembers(obj, predicate=inspect.ismethod)
        for name, obj_func in members:
            if obj_func.__func__ is func.__func__:
                return name
    raise ValueError(f'Function {func} is not an instance method in: {obj}')