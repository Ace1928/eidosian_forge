from xml.etree import ElementTree
from io import StringIO
from Bio.KEGG.KGML.KGML_pathway import Component, Entry, Graphics
from Bio.KEGG.KGML.KGML_pathway import Pathway, Reaction, Relation
def _parse_component(element, entry):
    new_component = Component(entry)
    for k, v in element.attrib.items():
        new_component.__setattr__(k, v)
    entry.add_component(new_component)