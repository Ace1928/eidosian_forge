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
def unbind_property(self, widget, name):
    """Unbind the handlers created by all the rules of the widget that set
        the name.

        This effectively clears all the rules of widget that take the form::

            name: rule

        For example:

        .. code-block:: python

            >>> w = Builder.load_string('''
            ... Widget:
            ...     height: self.width / 2. if self.disabled else self.width
            ...     x: self.y + 50
            ... ''')
            >>> w.size
            [100, 100]
            >>> w.pos
            [50, 0]
            >>> w.width = 500
            >>> w.size
            [500, 500]
            >>> Builder.unbind_property(w, 'height')
            >>> w.width = 222
            >>> w.size
            [222, 500]
            >>> w.y = 500
            >>> w.pos
            [550, 500]

        .. versionadded:: 1.9.1
        """
    uid = widget.uid
    if uid not in _handlers:
        return
    prop_handlers = _handlers[uid]
    if name not in prop_handlers:
        return
    for callbacks in prop_handlers[name]:
        for f, k, fn, bound_uid in callbacks:
            if fn is None:
                continue
            try:
                f.unbind_uid(k, bound_uid)
            except ReferenceError:
                pass
    del prop_handlers[name]
    if not prop_handlers:
        del _handlers[uid]