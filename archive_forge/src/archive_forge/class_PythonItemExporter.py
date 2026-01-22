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
class PythonItemExporter(BaseItemExporter):
    """This is a base class for item exporters that extends
    :class:`BaseItemExporter` with support for nested items.

    It serializes items to built-in Python types, so that any serialization
    library (e.g. :mod:`json` or msgpack_) can be used on top of it.

    .. _msgpack: https://pypi.org/project/msgpack/
    """

    def _configure(self, options, dont_fail=False):
        super()._configure(options, dont_fail)
        if not self.encoding:
            self.encoding = 'utf-8'

    def serialize_field(self, field, name, value):
        serializer = field.get('serializer', self._serialize_value)
        return serializer(value)

    def _serialize_value(self, value):
        if isinstance(value, Item):
            return self.export_item(value)
        if is_item(value):
            return dict(self._serialize_item(value))
        if is_listlike(value):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, (str, bytes)):
            return to_unicode(value, encoding=self.encoding)
        return value

    def _serialize_item(self, item):
        for key, value in ItemAdapter(item).items():
            yield (key, self._serialize_value(value))

    def export_item(self, item):
        result = dict(self._get_serialized_fields(item))
        return result