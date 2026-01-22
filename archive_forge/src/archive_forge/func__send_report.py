import logging
import socket
import sys
from importlib import import_module
from pprint import pformat
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.mail import MailSender
from scrapy.utils.engine import get_engine_status
def _send_report(self, rcpts, subject):
    """send notification mail with some additional useful info"""
    stats = self.crawler.stats
    s = f'Memory usage at engine startup : {stats.get_value('memusage/startup') / 1024 / 1024}M\r\n'
    s += f'Maximum memory usage          : {stats.get_value('memusage/max') / 1024 / 1024}M\r\n'
    s += f'Current memory usage          : {self.get_virtual_size() / 1024 / 1024}M\r\n'
    s += 'ENGINE STATUS ------------------------------------------------------- \r\n'
    s += '\r\n'
    s += pformat(get_engine_status(self.crawler.engine))
    s += '\r\n'
    self.mail.send(rcpts, subject, s)