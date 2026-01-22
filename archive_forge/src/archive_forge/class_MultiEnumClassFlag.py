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
class MultiEnumClassFlag(MultiFlag):
    """A multi_enum_class flag.

  See the __doc__ for MultiFlag for most behaviors of this class.  In addition,
  this class knows how to handle enum.Enum instances as values for this flag
  type.
  """

    def __init__(self, name, default, help_string, enum_class, case_sensitive=False, **args):
        p = _argument_parser.EnumClassParser(enum_class, case_sensitive=case_sensitive)
        g = _argument_parser.EnumClassListSerializer(list_sep=',', lowercase=not case_sensitive)
        super(MultiEnumClassFlag, self).__init__(p, g, name, default, help_string, **args)
        self.help = '<%s>: %s;\n    repeat this option to specify a list of values' % ('|'.join(p.member_names), help_string or '(no help available)')

    def _extra_xml_dom_elements(self, doc):
        elements = []
        for enum_value in self.parser.enum_class.__members__.keys():
            elements.append(_helpers.create_xml_dom_element(doc, 'enum_value', enum_value))
        return elements

    def _serialize_value_for_xml(self, value):
        """See base class."""
        if value is not None:
            value_serialized = self.serializer.serialize(value)
        else:
            value_serialized = ''
        return value_serialized