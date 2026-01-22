import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class ReadOnlySequentialNamedNodeMap(object):
    __slots__ = ('_seq',)

    def __init__(self, seq=()):
        self._seq = seq

    def __len__(self):
        return len(self._seq)

    def _get_length(self):
        return len(self._seq)

    def getNamedItem(self, name):
        for n in self._seq:
            if n.nodeName == name:
                return n

    def getNamedItemNS(self, namespaceURI, localName):
        for n in self._seq:
            if n.namespaceURI == namespaceURI and n.localName == localName:
                return n

    def __getitem__(self, name_or_tuple):
        if isinstance(name_or_tuple, tuple):
            node = self.getNamedItemNS(*name_or_tuple)
        else:
            node = self.getNamedItem(name_or_tuple)
        if node is None:
            raise KeyError(name_or_tuple)
        return node

    def item(self, index):
        if index < 0:
            return None
        try:
            return self._seq[index]
        except IndexError:
            return None

    def removeNamedItem(self, name):
        raise xml.dom.NoModificationAllowedErr('NamedNodeMap instance is read-only')

    def removeNamedItemNS(self, namespaceURI, localName):
        raise xml.dom.NoModificationAllowedErr('NamedNodeMap instance is read-only')

    def setNamedItem(self, node):
        raise xml.dom.NoModificationAllowedErr('NamedNodeMap instance is read-only')

    def setNamedItemNS(self, node):
        raise xml.dom.NoModificationAllowedErr('NamedNodeMap instance is read-only')

    def __getstate__(self):
        return [self._seq]

    def __setstate__(self, state):
        self._seq = state[0]