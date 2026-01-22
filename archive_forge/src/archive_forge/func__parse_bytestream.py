import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def _parse_bytestream(self, stream, options):
    import xml.dom.expatbuilder
    builder = xml.dom.expatbuilder.makeBuilder(options)
    return builder.parseFile(stream)