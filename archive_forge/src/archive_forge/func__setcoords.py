import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def _setcoords(self, value):
    clist = [int(e) for e in value.split(',')]
    self._coords = [tuple(clist[i:i + 2]) for i in range(0, len(clist), 2)]