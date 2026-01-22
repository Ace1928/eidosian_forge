import csv
import io
import marshal
import pickle
import pprint
from collections.abc import Mapping
from xml.sax.saxutils import XMLGenerator
from itemadapter import ItemAdapter, is_item
from scrapy.item import Item
from scrapy.utils.python import is_listlike, to_bytes, to_unicode
from scrapy.utils.serialize import ScrapyJSONEncoder
def _join_if_needed(self, value):
    if isinstance(value, (list, tuple)):
        try:
            return self._join_multivalued.join(value)
        except TypeError:
            pass
    return value