import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
class WindowMenu(Tk_.Menu, ListedWindow):
    """
    Menu with a postcommand which shows all listed windows.
    """

    def __init__(self, menubar):
        Tk_.Menu.__init__(self, menubar, name='window', postcommand=self.build_entries)

    def build_entries(self):
        if sys.platform == 'darwin':
            return
        self.delete(0, self.index(Tk_.END))
        for object in self.window_list:
            self.add_command(label=object.menu_title, command=object.bring_to_front)