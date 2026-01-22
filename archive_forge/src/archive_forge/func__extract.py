import sys
import os.path
from lxml import etree as _etree # due to validator __init__ signature
def _extract(self, element):
    """Extract embedded schematron schema from non-schematron host schema.
        This method will only be called by __init__ if the given schema document
        is not a schematron schema by itself.
        Must return a schematron schema document tree or None.
        """
    schematron = None
    if element.tag == _xml_schema_root:
        schematron = self._extract_xsd(element)
    elif element.nsmap[element.prefix] == RELAXNG_NS:
        schematron = self._extract_rng(element)
    return schematron