import subprocess
import sys
import time
from urllib.parse import urlencode
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.linkextractors import LinkExtractor
class _BenchSpider(scrapy.Spider):
    """A spider that follows all links"""
    name = 'follow'
    total = 10000
    show = 20
    baseurl = 'http://localhost:8998'
    link_extractor = LinkExtractor()

    def start_requests(self):
        qargs = {'total': self.total, 'show': self.show}
        url = f'{self.baseurl}?{urlencode(qargs, doseq=True)}'
        return [scrapy.Request(url, dont_filter=True)]

    def parse(self, response):
        for link in self.link_extractor.extract_links(response):
            yield scrapy.Request(link.url, callback=self.parse)