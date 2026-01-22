import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def add_extension_elements(self, items):
    for item in items:
        self.extension_elements.append(element_to_extension_element(item))