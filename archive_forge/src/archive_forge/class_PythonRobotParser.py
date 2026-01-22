import logging
import sys
from abc import ABCMeta, abstractmethod
from scrapy.utils.python import to_unicode
class PythonRobotParser(RobotParser):

    def __init__(self, robotstxt_body, spider):
        from urllib.robotparser import RobotFileParser
        self.spider = spider
        robotstxt_body = decode_robotstxt(robotstxt_body, spider, to_native_str_type=True)
        self.rp = RobotFileParser()
        self.rp.parse(robotstxt_body.splitlines())

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body):
        spider = None if not crawler else crawler.spider
        o = cls(robotstxt_body, spider)
        return o

    def allowed(self, url, user_agent):
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.rp.can_fetch(user_agent, url)