import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def add_relation(self, relation):
    """Add a Relation element to the pathway."""
    relation._pathway = self
    self._relations.add(relation)