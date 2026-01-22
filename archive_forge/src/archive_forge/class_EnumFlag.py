from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import functools
from absl._collections_abc import abc
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _helpers
import six
class EnumFlag(Flag):
    """Basic enum flag; its value can be any string from list of enum_values."""

    def __init__(self, name, default, help, enum_values, short_name=None, case_sensitive=True, **args):
        p = _argument_parser.EnumParser(enum_values, case_sensitive)
        g = _argument_parser.ArgumentSerializer()
        super(EnumFlag, self).__init__(p, g, name, default, help, short_name, **args)
        self.help = '<%s>: %s' % ('|'.join(enum_values), self.help)

    def _extra_xml_dom_elements(self, doc):
        elements = []
        for enum_value in self.parser.enum_values:
            elements.append(_helpers.create_xml_dom_element(doc, 'enum_value', enum_value))
        return elements