import logging
from datetime import datetime, timezone
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.serialize import ScrapyJSONEncoder
Log basic scraping stats periodically