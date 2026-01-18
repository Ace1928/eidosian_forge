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
send notification mail with some additional useful info