import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __getFileName(self, name, extension):
    postfix = 0
    while 1:
        if postfix == 0:
            filename = name + extension
        else:
            filename = name + '-' + str(postfix) + extension
        if extension == '.desktop':
            dir = 'applications'
        elif extension == '.directory':
            dir = 'desktop-directories'
        if not filename in self.filenames and (not os.path.isfile(os.path.join(xdg_data_dirs[0], dir, filename))):
            self.filenames.append(filename)
            break
        else:
            postfix += 1
    return filename