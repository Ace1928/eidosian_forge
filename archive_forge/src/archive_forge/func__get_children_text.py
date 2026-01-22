from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _get_children_text(parent, tag, construct=str):
    """Find child nodes by tag; pass each node's text through a constructor (PRIVATE).

    Returns an empty list if no matching child is found.
    """
    return [construct(child.text) for child in parent.findall(_ns(tag)) if child.text]