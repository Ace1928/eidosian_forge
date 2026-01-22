import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
class ComboBoxEntry(Gtk.ComboBox):

    def __init__(self, **kwds):
        Gtk.ComboBox.__init__(self, has_entry=True, **kwds)

    def set_text_column(self, text_column):
        self.set_entry_text_column(text_column)

    def get_text_column(self):
        return self.get_entry_text_column()