import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __getXmlMenu(self, path, create=True, element=None):
    if not element:
        element = self.tree
    if '/' in path:
        name, path = path.split('/', 1)
    else:
        name = path
        path = ''
    found = None
    for node in element.findall('Menu'):
        name_node = node.find('Name')
        if name_node.text == name:
            if path:
                found = self.__getXmlMenu(path, create, node)
            else:
                found = node
        if found:
            break
    if not found and create:
        node = self.__addXmlMenuElement(element, name)
        if path:
            found = self.__getXmlMenu(path, create, node)
        else:
            found = node
    return found