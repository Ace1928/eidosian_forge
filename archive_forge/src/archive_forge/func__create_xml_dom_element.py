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
def _create_xml_dom_element(self, doc, module_name, is_key=False):
    """Returns an XML element that contains this flag's information.

    This is information that is relevant to all flags (e.g., name,
    meaning, etc.).  If you defined a flag that has some other pieces of
    info, then please override _ExtraXMLInfo.

    Please do NOT override this method.

    Args:
      doc: minidom.Document, the DOM document it should create nodes from.
      module_name: str,, the name of the module that defines this flag.
      is_key: boolean, True iff this flag is key for main module.

    Returns:
      A minidom.Element instance.
    """
    element = doc.createElement('flag')
    if is_key:
        element.appendChild(_helpers.create_xml_dom_element(doc, 'key', 'yes'))
    element.appendChild(_helpers.create_xml_dom_element(doc, 'file', module_name))
    element.appendChild(_helpers.create_xml_dom_element(doc, 'name', self.name))
    if self.short_name:
        element.appendChild(_helpers.create_xml_dom_element(doc, 'short_name', self.short_name))
    if self.help:
        element.appendChild(_helpers.create_xml_dom_element(doc, 'meaning', self.help))
    if self.serializer and (not isinstance(self.default, str)):
        if self.default is not None:
            default_serialized = self.serializer.serialize(self.default)
        else:
            default_serialized = ''
    else:
        default_serialized = self.default
    element.appendChild(_helpers.create_xml_dom_element(doc, 'default', default_serialized))
    value_serialized = self._serialize_value_for_xml(self.value)
    element.appendChild(_helpers.create_xml_dom_element(doc, 'current', value_serialized))
    element.appendChild(_helpers.create_xml_dom_element(doc, 'type', self.flag_type()))
    for e in self._extra_xml_dom_elements(doc):
        element.appendChild(e)
    return element