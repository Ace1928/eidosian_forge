import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
class BaseGetter(object):

    def __init__(self, context):
        self.context = context

    def __getitem__(self, state):
        color = self.context.get_background_color(state)
        return Gdk.Color(red=int(color.red * 65535), green=int(color.green * 65535), blue=int(color.blue * 65535))