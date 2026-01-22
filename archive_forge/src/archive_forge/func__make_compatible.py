import functools
import logging
from collections import defaultdict
from inspect import signature
from warnings import warn
from twisted.internet.defer import Deferred, DeferredList
from twisted.python.failure import Failure
from scrapy.http.request import NO_CALLBACK
from scrapy.settings import Settings
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.defer import defer_result, mustbe_deferred
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
def _make_compatible(self):
    """Make overridable methods of MediaPipeline and subclasses backwards compatible"""
    methods = ['file_path', 'thumb_path', 'media_to_download', 'media_downloaded', 'file_downloaded', 'image_downloaded', 'get_images']
    for method_name in methods:
        method = getattr(self, method_name, None)
        if callable(method):
            setattr(self, method_name, self._compatible(method))