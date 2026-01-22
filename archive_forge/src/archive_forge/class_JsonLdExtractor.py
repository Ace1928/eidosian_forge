import json
import re
import jstyleson
import lxml.etree
from extruct.utils import parse_html
class JsonLdExtractor:
    _xp_jsonld = lxml.etree.XPath('descendant-or-self::script[@type="application/ld+json"]')

    def extract(self, htmlstring, base_url=None, encoding='UTF-8'):
        tree = parse_html(htmlstring, encoding=encoding)
        return self.extract_items(tree, base_url=base_url)

    def extract_items(self, document, base_url=None):
        return [item for items in map(self._extract_items, self._xp_jsonld(document)) if items for item in items if item]

    def _extract_items(self, node):
        script = node.xpath('string()')
        try:
            data = json.loads(script, strict=False)
        except ValueError:
            data = jstyleson.loads(HTML_OR_JS_COMMENTLINE.sub('', script), strict=False)
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            yield data