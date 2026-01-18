import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def to_string_force_namespace(self, nspair):
    elem = self._to_element_tree()
    self.set_prefixes(elem, nspair)
    return ElementTree.tostring(elem, encoding='UTF-8')