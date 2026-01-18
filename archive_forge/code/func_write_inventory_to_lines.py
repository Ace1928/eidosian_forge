from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
def write_inventory_to_lines(self, inv):
    """Return a list of lines with the encoded inventory."""
    return self.write_inventory(inv, None)