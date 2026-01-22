from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
def _write_element(self, elt, f):
    ElementTree(elt).write(f, 'utf-8')
    f.write(b'\n')