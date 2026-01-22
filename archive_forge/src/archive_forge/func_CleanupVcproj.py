import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def CleanupVcproj(node):
    """For each sub node, we call recursively this function."""
    for sub_node in node.childNodes:
        AbsoluteNode(sub_node)
        CleanupVcproj(sub_node)
    for sub_node in node.childNodes:
        if sub_node.nodeType == Node.TEXT_NODE:
            sub_node.data = sub_node.data.replace('\r', '')
            sub_node.data = sub_node.data.replace('\n', '')
            sub_node.data = sub_node.data.rstrip()
    if node.attributes:
        for name, value in node.attributes.items():
            sorted_list = sorted(value.split(';'))
            unique_list = []
            for i in sorted_list:
                if not unique_list.count(i):
                    unique_list.append(i)
            node.setAttribute(name, ';'.join(unique_list))
            if not value:
                node.removeAttribute(name)
    if node.childNodes:
        node.normalize()
    node_array = []
    while node.childNodes and node.childNodes[0]:
        current = node.childNodes[0]
        node.removeChild(current)
        if current.nodeName == 'Filter':
            node_array.extend(FlattenFilter(current))
        else:
            node_array.append(current)
    node_array.sort(CmpNode())
    for new_node in node_array:
        if new_node.nodeName == 'Tool':
            if new_node.attributes and new_node.attributes.length == 1:
                continue
        if new_node.nodeName == 'UserMacro':
            continue
        node.appendChild(new_node)