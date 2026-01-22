from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _local(tag):
    """Extract the local tag from a namespaced tag name (PRIVATE)."""
    if tag[0] == '{':
        return tag[tag.index('}') + 1:]
    return tag