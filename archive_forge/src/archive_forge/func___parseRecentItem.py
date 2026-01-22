import xml.dom.minidom, xml.sax.saxutils
import os, time, fcntl
from xdg.Exceptions import ParsingError
def __parseRecentItem(self, item):
    recent = RecentFile()
    self.RecentFiles.append(recent)
    for attribute in item.childNodes:
        if attribute.nodeType == xml.dom.Node.ELEMENT_NODE:
            if attribute.tagName == 'URI':
                recent.URI = attribute.childNodes[0].nodeValue
            elif attribute.tagName == 'Mime-Type':
                recent.MimeType = attribute.childNodes[0].nodeValue
            elif attribute.tagName == 'Timestamp':
                recent.Timestamp = int(attribute.childNodes[0].nodeValue)
            elif attribute.tagName == 'Private':
                recent.Prviate = True
            elif attribute.tagName == 'Groups':
                for group in attribute.childNodes:
                    if group.nodeType == xml.dom.Node.ELEMENT_NODE:
                        if group.tagName == 'Group':
                            recent.Groups.append(group.childNodes[0].nodeValue)