from scrapy.exceptions import NotConfigured, NotSupported
from scrapy.selector import Selector
from scrapy.spiders import Spider
from scrapy.utils.iterators import csviter, xmliter_lxml
from scrapy.utils.spider import iterate_spider_output
def _register_namespaces(self, selector):
    for prefix, uri in self.namespaces:
        selector.register_namespace(prefix, uri)