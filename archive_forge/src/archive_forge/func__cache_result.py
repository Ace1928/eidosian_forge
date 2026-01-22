from typing import Any
from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import (
from zope.interface.declarations import implementer, provider
from scrapy.utils.datatypes import LocalCache
def _cache_result(self, result, name):
    dnscache[name] = result
    return result