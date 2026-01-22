import logging
from scrapy import signals
from scrapy.exceptions import NotConfigured
def _max_delay(self, spider):
    return self.crawler.settings.getfloat('AUTOTHROTTLE_MAX_DELAY')