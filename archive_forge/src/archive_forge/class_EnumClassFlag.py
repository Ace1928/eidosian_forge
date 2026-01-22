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
class EnumClassFlag(Flag):
    """Basic enum flag; its value is an enum class's member."""

    def __init__(self, name, default, help, enum_class, short_name=None, case_sensitive=False, **args):
        p = _argument_parser.EnumClassParser(enum_class, case_sensitive=case_sensitive)
        g = _argument_parser.EnumClassSerializer(lowercase=not case_sensitive)
        super(EnumClassFlag, self).__init__(p, g, name, default, help, short_name, **args)
        self.help = '<%s>: %s' % ('|'.join(p.member_names), self.help)

    def _extra_xml_dom_elements(self, doc):
        elements = []
        for enum_value in self.parser.enum_class.__members__.keys():
            elements.append(_helpers.create_xml_dom_element(doc, 'enum_value', enum_value))
        return elements