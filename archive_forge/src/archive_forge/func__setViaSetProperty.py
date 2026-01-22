import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _setViaSetProperty(self, widget, prop):
    prop_value = self.convert(prop, widget)
    if prop_value is not None:
        prop_name = prop.attrib['name']
        if prop[0].tag == 'cursorShape':
            widget.viewport().setProperty(prop_name, prop_value)
        else:
            widget.setProperty(prop_name, prop_value)