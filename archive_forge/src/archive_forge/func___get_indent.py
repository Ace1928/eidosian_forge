import codecs
from xml.sax.saxutils import escape, quoteattr
def __get_indent(self):
    return '  ' * len(self.element_stack)