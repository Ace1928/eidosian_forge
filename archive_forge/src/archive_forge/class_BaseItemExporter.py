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
class BaseItemExporter:

    def __init__(self, *, dont_fail=False, **kwargs):
        self._kwargs = kwargs
        self._configure(kwargs, dont_fail=dont_fail)

    def _configure(self, options, dont_fail=False):
        """Configure the exporter by popping options from the ``options`` dict.
        If dont_fail is set, it won't raise an exception on unexpected options
        (useful for using with keyword arguments in subclasses ``__init__`` methods)
        """
        self.encoding = options.pop('encoding', None)
        self.fields_to_export = options.pop('fields_to_export', None)
        self.export_empty_fields = options.pop('export_empty_fields', False)
        self.indent = options.pop('indent', None)
        if not dont_fail and options:
            raise TypeError(f'Unexpected options: {', '.join(options.keys())}')

    def export_item(self, item):
        raise NotImplementedError

    def serialize_field(self, field, name, value):
        serializer = field.get('serializer', lambda x: x)
        return serializer(value)

    def start_exporting(self):
        pass

    def finish_exporting(self):
        pass

    def _get_serialized_fields(self, item, default_value=None, include_empty=None):
        """Return the fields to export as an iterable of tuples
        (name, serialized_value)
        """
        item = ItemAdapter(item)
        if include_empty is None:
            include_empty = self.export_empty_fields
        if self.fields_to_export is None:
            if include_empty:
                field_iter = item.field_names()
            else:
                field_iter = item.keys()
        elif isinstance(self.fields_to_export, Mapping):
            if include_empty:
                field_iter = self.fields_to_export.items()
            else:
                field_iter = ((x, y) for x, y in self.fields_to_export.items() if x in item)
        elif include_empty:
            field_iter = self.fields_to_export
        else:
            field_iter = (x for x in self.fields_to_export if x in item)
        for field_name in field_iter:
            if isinstance(field_name, str):
                item_field, output_field = (field_name, field_name)
            else:
                item_field, output_field = field_name
            if item_field in item:
                field_meta = item.get_field_meta(item_field)
                value = self.serialize_field(field_meta, output_field, item[item_field])
            else:
                value = default_value
            yield (output_field, value)