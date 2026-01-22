import logging
import pprint
from typing import TYPE_CHECKING, Any, Dict, Optional
from scrapy import Spider
def _persist_stats(self, stats: StatsT, spider: Spider) -> None:
    self.spider_stats[spider.name] = stats