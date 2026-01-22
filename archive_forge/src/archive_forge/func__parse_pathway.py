from xml.etree import ElementTree
from io import StringIO
from Bio.KEGG.KGML.KGML_pathway import Component, Entry, Graphics
from Bio.KEGG.KGML.KGML_pathway import Pathway, Reaction, Relation
def _parse_pathway(attrib):
    for k, v in attrib.items():
        self.pathway.__setattr__(k, v)