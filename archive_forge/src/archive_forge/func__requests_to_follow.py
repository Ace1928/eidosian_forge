import copy
from typing import AsyncIterable, Awaitable, Sequence
from scrapy.http import HtmlResponse, Request, Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Spider
from scrapy.utils.asyncgen import collect_asyncgen
from scrapy.utils.spider import iterate_spider_output
def _requests_to_follow(self, response):
    if not isinstance(response, HtmlResponse):
        return
    seen = set()
    for rule_index, rule in enumerate(self._rules):
        links = [lnk for lnk in rule.link_extractor.extract_links(response) if lnk not in seen]
        for link in rule.process_links(links):
            seen.add(link)
            request = self._build_request(rule_index, link)
            yield rule.process_request(request, response)