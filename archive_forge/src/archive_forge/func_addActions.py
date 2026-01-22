import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def addActions(self):
    for widget, action_name in self.actions:
        if action_name == 'separator':
            widget.addSeparator()
        else:
            DEBUG('add action %s to %s', action_name, widget.objectName())
            action_obj = getattr(self.toplevelWidget, action_name)
            if isinstance(action_obj, QtWidgets.QMenu):
                widget.addAction(action_obj.menuAction())
            elif not isinstance(action_obj, QtWidgets.QActionGroup):
                widget.addAction(action_obj)