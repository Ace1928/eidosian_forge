import json, sys
from xml.dom import minidom
import plistlib
import logging
def xmlToDict(self, xmlNode):
    if xmlNode.nodeName == '#document':
        node = {xmlNode.firstChild.nodeName: {}}
        node[xmlNode.firstChild.nodeName] = self.xmlToDict(xmlNode.firstChild)
        return node
    node = {}
    curr = node
    if xmlNode.attributes:
        for name, value in xmlNode.attributes.items():
            curr[name] = value
    for n in xmlNode.childNodes:
        if n.nodeType == n.TEXT_NODE:
            curr['__TEXT__'] = n.data
            continue
        if not n.nodeName in curr:
            curr[n.nodeName] = []
        if len(xmlNode.getElementsByTagName(n.nodeName)) > 1:
            curr[n.nodeName].append(self.xmlToDict(n))
        else:
            curr[n.nodeName] = self.xmlToDict(n)
    return node