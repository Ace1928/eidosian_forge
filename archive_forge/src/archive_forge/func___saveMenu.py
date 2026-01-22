import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __saveMenu(self):
    if not os.path.isdir(os.path.dirname(self.filename)):
        os.makedirs(os.path.dirname(self.filename))
    self.tree.write(self.filename, encoding='utf-8')