import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __deleteFile(self, filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    try:
        self.filenames.remove(filename)
    except ValueError:
        pass