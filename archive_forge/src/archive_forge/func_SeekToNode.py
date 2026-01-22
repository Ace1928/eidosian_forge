import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def SeekToNode(node1, child2):
    if child2.nodeType == Node.TEXT_NODE:
        return None
    current_name = child2.getAttribute('Name')
    if not current_name:
        return None
    for sub_node in node1.childNodes:
        if sub_node.nodeName == child2.nodeName:
            name = sub_node.getAttribute('Name')
            if name == current_name:
                return sub_node
    return None