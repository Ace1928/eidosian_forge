import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def add_substrate(self, substrate_id):
    """Add a substrate, identified by its node ID, to the reaction."""
    if self._pathway is not None:
        if int(substrate_id) not in self._pathway.entries:
            raise ValueError("Couldn't add substrate, no node ID %d in Pathway" % int(substrate_id))
    self._substrates.add(substrate_id)