import logging
import re
from typing import TYPE_CHECKING, Any
from scrapy.http import Request, XmlResponse
from scrapy.spiders import Spider
from scrapy.utils._compression import _DecompressionMaxSizeExceeded
from scrapy.utils.gz import gunzip, gzip_magic_number
from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
def _parse_sitemap(self, response):
    if response.url.endswith('/robots.txt'):
        for url in sitemap_urls_from_robots(response.text, base_url=response.url):
            yield Request(url, callback=self._parse_sitemap)
    else:
        body = self._get_sitemap_body(response)
        if body is None:
            logger.warning('Ignoring invalid sitemap: %(response)s', {'response': response}, extra={'spider': self})
            return
        s = Sitemap(body)
        it = self.sitemap_filter(s)
        if s.type == 'sitemapindex':
            for loc in iterloc(it, self.sitemap_alternate_links):
                if any((x.search(loc) for x in self._follow)):
                    yield Request(loc, callback=self._parse_sitemap)
        elif s.type == 'urlset':
            for loc in iterloc(it, self.sitemap_alternate_links):
                for r, c in self._cbs:
                    if r.search(loc):
                        yield Request(loc, callback=c)
                        break