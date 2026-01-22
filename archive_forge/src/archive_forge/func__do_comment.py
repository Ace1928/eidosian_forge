import string
from xml.dom import Node
def _do_comment(self, node):
    """_do_comment(self, node) -> None
        Process a comment node. Render a leading or trailing #xA if the
        document order of the comment is greater or lesser (respectively)
        than the document element.
        """
    if not _in_subset(self.subset, node):
        return
    if self.comments:
        W = self.write
        if self.documentOrder == _GreaterElement:
            W('\n')
        W('<!--')
        W(node.data)
        W('-->')
        if self.documentOrder == _LesserElement:
            W('\n')