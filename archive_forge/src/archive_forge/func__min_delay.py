import logging
from scrapy import signals
from scrapy.exceptions import NotConfigured
def _min_delay(self, spider):
    s = self.crawler.settings
    return getattr(spider, 'download_delay', s.getfloat('DOWNLOAD_DELAY'))