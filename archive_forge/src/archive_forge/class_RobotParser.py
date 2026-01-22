import logging
import sys
from abc import ABCMeta, abstractmethod
from scrapy.utils.python import to_unicode
class RobotParser(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_crawler(cls, crawler, robotstxt_body):
        """Parse the content of a robots.txt_ file as bytes. This must be a class method.
        It must return a new instance of the parser backend.

        :param crawler: crawler which made the request
        :type crawler: :class:`~scrapy.crawler.Crawler` instance

        :param robotstxt_body: content of a robots.txt_ file.
        :type robotstxt_body: bytes
        """
        pass

    @abstractmethod
    def allowed(self, url, user_agent):
        """Return ``True`` if  ``user_agent`` is allowed to crawl ``url``, otherwise return ``False``.

        :param url: Absolute URL
        :type url: str

        :param user_agent: User agent
        :type user_agent: str
        """
        pass