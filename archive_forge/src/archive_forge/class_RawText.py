import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
class RawText(microdom.Text):
    """This is an evil and horrible speed hack. Basically, if you have a big
    chunk of XML that you want to insert into the DOM, but you don't want to
    incur the cost of parsing it, you can construct one of these and insert it
    into the DOM. This will most certainly only work with microdom as the API
    for converting nodes to xml is different in every DOM implementation.

    This could be improved by making this class a Lazy parser, so if you
    inserted this into the DOM and then later actually tried to mutate this
    node, it would be parsed then.
    """

    def writexml(self, writer, indent='', addindent='', newl='', strip=0, nsprefixes=None, namespace=None):
        writer.write(f'{indent}{self.data}{newl}')