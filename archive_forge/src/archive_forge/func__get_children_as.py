from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _get_children_as(parent, tag, construct):
    """Find child nodes by tag; pass each through a constructor (PRIVATE).

    Returns an empty list if no matching child is found.
    """
    return [construct(child) for child in parent.findall(_ns(tag))]