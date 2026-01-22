from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class ListParser(BaseListParser):
    """Parser for a comma-separated list of strings."""

    def __init__(self):
        super(ListParser, self).__init__(',', 'comma')

    def parse(self, argument):
        """Parses argument as comma-separated list of strings."""
        if isinstance(argument, list):
            return argument
        elif not argument:
            return []
        else:
            try:
                return [s.strip() for s in list(csv.reader([argument], strict=True))[0]]
            except csv.Error as e:
                raise ValueError('Unable to parse the value %r as a %s: %s' % (argument, self.flag_type(), e))

    def _custom_xml_dom_elements(self, doc):
        elements = super(ListParser, self)._custom_xml_dom_elements(doc)
        elements.append(_helpers.create_xml_dom_element(doc, 'list_separator', repr(',')))
        return elements