import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def _get_opener(self):
    try:
        return self._opener
    except AttributeError:
        self._opener = self._create_opener()
        return self._opener