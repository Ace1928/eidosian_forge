from __future__ import annotations
import logging
import pprint
import signal
import warnings
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Set, Type, Union, cast
from twisted.internet.defer import (
from zope.interface.exceptions import DoesNotImplement
from zope.interface.verify import verifyClass
from scrapy import Spider, signals
from scrapy.addons import AddonManager
from scrapy.core.engine import ExecutionEngine
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.extension import ExtensionManager
from scrapy.interfaces import ISpiderLoader
from scrapy.logformatter import LogFormatter
from scrapy.settings import BaseSettings, Settings, overridden_settings
from scrapy.signalmanager import SignalManager
from scrapy.statscollectors import StatsCollector
from scrapy.utils.log import (
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.ossignal import install_shutdown_handlers, signal_names
from scrapy.utils.reactor import (
class CrawlerProcess(CrawlerRunner):
    """
    A class to run multiple scrapy crawlers in a process simultaneously.

    This class extends :class:`~scrapy.crawler.CrawlerRunner` by adding support
    for starting a :mod:`~twisted.internet.reactor` and handling shutdown
    signals, like the keyboard interrupt command Ctrl-C. It also configures
    top-level logging.

    This utility should be a better fit than
    :class:`~scrapy.crawler.CrawlerRunner` if you aren't running another
    :mod:`~twisted.internet.reactor` within your application.

    The CrawlerProcess object must be instantiated with a
    :class:`~scrapy.settings.Settings` object.

    :param install_root_handler: whether to install root logging handler
        (default: True)

    This class shouldn't be needed (since Scrapy is responsible of using it
    accordingly) unless writing scripts that manually handle the crawling
    process. See :ref:`run-from-script` for an example.
    """

    def __init__(self, settings: Union[Dict[str, Any], Settings, None]=None, install_root_handler: bool=True):
        super().__init__(settings)
        configure_logging(self.settings, install_root_handler)
        log_scrapy_info(self.settings)
        self._initialized_reactor = False

    def _signal_shutdown(self, signum: int, _: Any) -> None:
        from twisted.internet import reactor
        install_shutdown_handlers(self._signal_kill)
        signame = signal_names[signum]
        logger.info('Received %(signame)s, shutting down gracefully. Send again to force ', {'signame': signame})
        reactor.callFromThread(self._graceful_stop_reactor)

    def _signal_kill(self, signum: int, _: Any) -> None:
        from twisted.internet import reactor
        install_shutdown_handlers(signal.SIG_IGN)
        signame = signal_names[signum]
        logger.info('Received %(signame)s twice, forcing unclean shutdown', {'signame': signame})
        reactor.callFromThread(self._stop_reactor)

    def _create_crawler(self, spidercls: Union[Type[Spider], str]) -> Crawler:
        if isinstance(spidercls, str):
            spidercls = self.spider_loader.load(spidercls)
        init_reactor = not self._initialized_reactor
        self._initialized_reactor = True
        return Crawler(cast(Type[Spider], spidercls), self.settings, init_reactor=init_reactor)

    def start(self, stop_after_crawl: bool=True, install_signal_handlers: bool=True) -> None:
        """
        This method starts a :mod:`~twisted.internet.reactor`, adjusts its pool
        size to :setting:`REACTOR_THREADPOOL_MAXSIZE`, and installs a DNS cache
        based on :setting:`DNSCACHE_ENABLED` and :setting:`DNSCACHE_SIZE`.

        If ``stop_after_crawl`` is True, the reactor will be stopped after all
        crawlers have finished, using :meth:`join`.

        :param bool stop_after_crawl: stop or not the reactor when all
            crawlers have finished

        :param bool install_signal_handlers: whether to install the OS signal
            handlers from Twisted and Scrapy (default: True)
        """
        from twisted.internet import reactor
        if stop_after_crawl:
            d = self.join()
            if d.called:
                return
            d.addBoth(self._stop_reactor)
        resolver_class = load_object(self.settings['DNS_RESOLVER'])
        resolver = create_instance(resolver_class, self.settings, self, reactor=reactor)
        resolver.install_on_reactor()
        tp = reactor.getThreadPool()
        tp.adjustPoolsize(maxthreads=self.settings.getint('REACTOR_THREADPOOL_MAXSIZE'))
        reactor.addSystemEventTrigger('before', 'shutdown', self.stop)
        if install_signal_handlers:
            reactor.addSystemEventTrigger('after', 'startup', install_shutdown_handlers, self._signal_shutdown)
        reactor.run(installSignalHandlers=install_signal_handlers)

    def _graceful_stop_reactor(self) -> Deferred:
        d = self.stop()
        d.addBoth(self._stop_reactor)
        return d

    def _stop_reactor(self, _: Any=None) -> None:
        from twisted.internet import reactor
        try:
            reactor.stop()
        except RuntimeError:
            pass