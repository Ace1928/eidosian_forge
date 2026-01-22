import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def _add_members_to_element_tree(self, tree):
    for member_name in self._get_all_c_children_with_order():
        member = getattr(self, member_name)
        if member is None:
            pass
        elif isinstance(member, list):
            for instance in member:
                instance.become_child_element_of(tree)
        else:
            member.become_child_element_of(tree)
    for xml_attribute, attribute_info in iter(self.__class__.c_attributes.items()):
        member_name, member_type, required = attribute_info
        member = getattr(self, member_name)
        if member is not None:
            tree.attrib[xml_attribute] = member
    ExtensionContainer._add_members_to_element_tree(self, tree)