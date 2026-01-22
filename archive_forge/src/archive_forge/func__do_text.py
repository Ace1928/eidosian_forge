import string
from xml.dom import Node
def _do_text(self, node):
    """_do_text(self, node) -> None
        Process a text or CDATA node.  Render various special characters
        as their C14N entity representations."""
    if not _in_subset(self.subset, node):
        return
    s = string.replace(node.data, '&', '&amp;')
    s = string.replace(s, '<', '&lt;')
    s = string.replace(s, '>', '&gt;')
    s = string.replace(s, '\r', '&#xD;')
    if s:
        self.write(s)