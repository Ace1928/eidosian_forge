import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def transfer_to_element_tree(self):
    if self.tag is None:
        return None
    element_tree = ElementTree.Element('')
    if self.namespace is not None:
        element_tree.tag = f'{{{self.namespace}}}{self.tag}'
    else:
        element_tree.tag = self.tag
    for key, value in iter(self.attributes.items()):
        element_tree.attrib[key] = value
    for child in self.children:
        child.become_child_element_of(element_tree)
    element_tree.text = self.text
    return element_tree