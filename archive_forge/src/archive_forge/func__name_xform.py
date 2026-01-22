import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def _name_xform(name):
    return name.lower().replace('-', '_')