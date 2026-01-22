import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def add_product(self, product_id):
    """Add a product, identified by its node ID, to the reaction."""
    if self._pathway is not None:
        if int(product_id) not in self._pathway.entries:
            raise ValueError("Couldn't add product, no node ID %d in Pathway" % product_id)
    self._products.add(int(product_id))