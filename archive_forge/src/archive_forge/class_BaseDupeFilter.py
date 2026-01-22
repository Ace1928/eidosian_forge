from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set
from warnings import warn
from twisted.internet.defer import Deferred
from scrapy.http.request import Request
from scrapy.settings import BaseSettings
from scrapy.spiders import Spider
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.job import job_dir
from scrapy.utils.request import (
class BaseDupeFilter:

    @classmethod
    def from_settings(cls, settings: BaseSettings) -> Self:
        return cls()

    def request_seen(self, request: Request) -> bool:
        return False

    def open(self) -> Optional[Deferred]:
        pass

    def close(self, reason: str) -> Optional[Deferred]:
        pass

    def log(self, request: Request, spider: Spider) -> None:
        """Log that a request has been filtered"""
        pass