from html.parser import HTMLParser
from itertools import zip_longest
def _write_children(self, doc):
    for child, next_child in zip_longest(self.children, self.children[1:]):
        if isinstance(child, TagNode) and next_child is not None:
            child.write(doc, next_child)
        else:
            child.write(doc)