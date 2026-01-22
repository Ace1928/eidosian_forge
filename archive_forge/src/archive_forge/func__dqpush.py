from __future__ import annotations
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, cast
from twisted.internet.defer import Deferred
from scrapy.crawler import Crawler
from scrapy.dupefilters import BaseDupeFilter
from scrapy.http.request import Request
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.job import job_dir
from scrapy.utils.misc import create_instance, load_object
def _dqpush(self, request: Request) -> bool:
    if self.dqs is None:
        return False
    try:
        self.dqs.push(request)
    except ValueError as e:
        if self.logunser:
            msg = 'Unable to serialize request: %(request)s - reason: %(reason)s - no more unserializable requests will be logged (stats being collected)'
            logger.warning(msg, {'request': request, 'reason': e}, exc_info=True, extra={'spider': self.spider})
            self.logunser = False
        assert self.stats is not None
        self.stats.inc_value('scheduler/unserializable', spider=self.spider)
        return False
    else:
        return True