from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
XML Element -> Revision object