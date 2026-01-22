from scrapy.exceptions import NotConfigured, NotSupported
from scrapy.selector import Selector
from scrapy.spiders import Spider
from scrapy.utils.iterators import csviter, xmliter_lxml
from scrapy.utils.spider import iterate_spider_output
def adapt_response(self, response):
    """This method has the same purpose as the one in XMLFeedSpider"""
    return response