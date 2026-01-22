from html.parser import HTMLParser
from itertools import zip_longest
class DocStringParser(HTMLParser):
    """
    A simple HTML parser.  Focused on converting the subset of HTML
    that appears in the documentation strings of the JSON models into
    simple ReST format.
    """

    def __init__(self, doc):
        self.tree = None
        self.doc = doc
        super().__init__()

    def reset(self):
        HTMLParser.reset(self)
        self.tree = HTMLTree(self.doc)

    def feed(self, data):
        super().feed(data)
        self.tree.write()
        self.tree = HTMLTree(self.doc)

    def close(self):
        super().close()
        self.tree.write()
        self.tree = HTMLTree(self.doc)

    def handle_starttag(self, tag, attrs):
        self.tree.add_tag(tag, attrs=attrs)

    def handle_endtag(self, tag):
        self.tree.add_tag(tag, is_start=False)

    def handle_data(self, data):
        self.tree.add_data(data)