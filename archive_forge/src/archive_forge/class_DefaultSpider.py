from __future__ import annotations
import inspect
import logging
from types import CoroutineType, ModuleType
from typing import (
from twisted.internet.defer import Deferred
from scrapy import Request
from scrapy.spiders import Spider
from scrapy.utils.defer import deferred_from_coro
from scrapy.utils.misc import arg_to_iter
class DefaultSpider(Spider):
    name = 'default'