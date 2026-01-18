from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
def write_revision_to_string(self, rev):
    return tostring(self._pack_revision(rev)) + b'\n'