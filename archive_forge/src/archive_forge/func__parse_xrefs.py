import re
from xml.etree import ElementTree
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_xrefs(self, root_entry_elem):
    """Parse xrefs (PRIVATE)."""
    xrefs = []
    if root_entry_elem is not None:
        xrefs.append('IPR:' + root_entry_elem.attrib['ac'])
    if root_entry_elem is not None:
        xref_elems = []
        xref_elems = xref_elems + root_entry_elem.findall(self.NS + 'go-xref')
        xref_elems = xref_elems + root_entry_elem.findall(self.NS + 'pathway-xref')
        for entry in xref_elems:
            xref = entry.attrib['id']
            if ':' not in xref:
                xref = entry.attrib['db'] + ':' + xref
            xrefs.append(xref)
    return xrefs