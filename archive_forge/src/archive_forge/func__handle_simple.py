from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _handle_simple(tag):
    """Handle to serialize simple nodes (PRIVATE)."""

    def wrapped(self, obj):
        """Wrap node as element."""
        elem = ElementTree.Element(tag)
        elem.text = _serialize(obj)
        return elem
    wrapped.__doc__ = f'Serialize a simple {tag} node.'
    return wrapped