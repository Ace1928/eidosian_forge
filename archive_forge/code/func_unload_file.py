import sys
from os import environ
from os.path import join
from copy import copy
from types import CodeType
from functools import partial
from kivy.factory import Factory
from kivy.lang.parser import (
from kivy.logger import Logger
from kivy.utils import QueryDict
from kivy.cache import Cache
from kivy import kivy_data_dir
from kivy.context import register_context
from kivy.resources import resource_find
from kivy._event import Observable, EventDispatcher
def unload_file(self, filename):
    """Unload all rules associated with a previously imported file.

        .. versionadded:: 1.0.8

        .. warning::

            This will not remove rules or templates already applied/used on
            current widgets. It will only effect the next widgets creation or
            template invocation.
        """
    filename = resource_find(filename) or filename
    self.rules = [x for x in self.rules if x[1].ctx.filename != filename]
    self._clear_matchcache()
    templates = {}
    for x, y in self.templates.items():
        if y[2] != filename:
            templates[x] = y
    self.templates = templates
    if filename in self.files:
        self.files.remove(filename)
    Factory.unregister_from_filename(filename)