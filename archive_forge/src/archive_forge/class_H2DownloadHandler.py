from time import time
from typing import Optional, Type, TypeVar
from urllib.parse import urldefrag
from twisted.internet.base import DelayedCall
from twisted.internet.defer import Deferred
from twisted.internet.error import TimeoutError
from twisted.web.client import URI
from scrapy.core.downloader.contextfactory import load_context_factory_from_settings
from scrapy.core.downloader.webclient import _parse
from scrapy.core.http2.agent import H2Agent, H2ConnectionPool, ScrapyProxyH2Agent
from scrapy.crawler import Crawler
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.python import to_bytes
class H2DownloadHandler:

    def __init__(self, settings: Settings, crawler: Optional[Crawler]=None):
        self._crawler = crawler
        from twisted.internet import reactor
        self._pool = H2ConnectionPool(reactor, settings)
        self._context_factory = load_context_factory_from_settings(settings, crawler)

    @classmethod
    def from_crawler(cls: Type[H2DownloadHandlerOrSubclass], crawler: Crawler) -> H2DownloadHandlerOrSubclass:
        return cls(crawler.settings, crawler)

    def download_request(self, request: Request, spider: Spider) -> Deferred:
        agent = ScrapyH2Agent(context_factory=self._context_factory, pool=self._pool, crawler=self._crawler)
        return agent.download_request(request, spider)

    def close(self) -> None:
        self._pool.close_connections()