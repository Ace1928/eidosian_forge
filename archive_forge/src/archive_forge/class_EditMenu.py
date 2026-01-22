import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
class EditMenu(Tk_.Menu):
    """Edit Menu cascade containing Cut, Copy, Paste and Delete. To use,
    provide a callback function which returns a dict specifying which
    editing functions should be enabled.  The keys should be chosen
    from the list ['Cut', 'Copy', 'Paste, 'Delete'] and the values
    should be functions to be call for the corresponding actions.  If
    a key is missing, the corresponding menu entry will be disabled.

    """
    entries = ['Cut', 'Copy', 'Paste', 'Delete']

    def __init__(self, menubar, callback):
        Tk_.Menu.__init__(self, menubar, name='snappyedit', postcommand=self.configure)
        self.get_actions = callback
        self.add_entry('Cut', lambda event=None: self.actions['Cut']())
        self.add_entry('Copy', lambda event=None: self.actions['Copy']())
        self.add_entry('Paste', lambda event=None: self.actions['Paste']())
        self.add_entry('Delete', lambda event=None: self.actions['Delete']())
        self.actions = {}

    def add_entry(self, label, command):
        accelerator = scut.get(label, '')
        self.add_command(label=label, accelerator=accelerator, command=command, state='disabled')

    def configure(self):
        """Called before the menu is opened."""
        self.actions = self.get_actions()
        for entry in self.entries:
            if self.actions.get(entry, None):
                self.entryconfig(entry, state='normal')
            else:
                self.entryconfig(entry, state='disabled')