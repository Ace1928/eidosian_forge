import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __addXmlMenuElement(self, element, name):
    menu_node = etree.SubElement('Menu', element)
    name_node = etree.SubElement('Name', menu_node)
    name_node.text = name
    return menu_node