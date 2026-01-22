import logging
from scrapy import signals
from scrapy.exceptions import NotConfigured
def _start_delay(self, spider):
    return max(self.mindelay, self.crawler.settings.getfloat('AUTOTHROTTLE_START_DELAY'))