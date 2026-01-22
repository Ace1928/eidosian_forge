from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _split_namespace(tag):
    """Split a tag into namespace and local tag strings (PRIVATE)."""
    try:
        return tag[1:].split('}', 1)
    except ValueError:
        return ('', tag)