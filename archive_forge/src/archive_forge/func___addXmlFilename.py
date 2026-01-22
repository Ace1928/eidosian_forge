import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __addXmlFilename(self, element, filename, type_='Include'):
    includes = element.findall('Include')
    excludes = element.findall('Exclude')
    rules = includes + excludes
    for rule in rules:
        if rule[0].tag == 'Filename' and rule[0].text == filename:
            element.remove(rule)
    node = etree.SubElement(type_, element)
    self.__addXmlTextElement(node, 'Filename', filename)
    return node