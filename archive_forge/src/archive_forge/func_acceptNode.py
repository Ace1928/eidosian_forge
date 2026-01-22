import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def acceptNode(self, element):
    return self.FILTER_ACCEPT