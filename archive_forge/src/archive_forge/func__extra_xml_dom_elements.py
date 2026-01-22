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
def _extra_xml_dom_elements(self, doc):
    elements = []
    for enum_value in self.parser.enum_class.__members__.keys():
        elements.append(_helpers.create_xml_dom_element(doc, 'enum_value', enum_value))
    return elements