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
def unbind_widget(self, uid):
    """Unbind all the handlers created by the KV rules of the
        widget. The :attr:`kivy.uix.widget.Widget.uid` is passed here
        instead of the widget itself, because Builder is using it in the
        widget destructor.

        This effectively clears all the KV rules associated with this widget.
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
            >>> Builder.unbind_widget(w.uid)
            >>> w.width = 222
            >>> w.y = 500
            >>> w.size
            [222, 500]
            >>> w.pos
            [50, 500]

        .. versionadded:: 1.7.2
        """
    if uid not in _handlers:
        return
    for prop_callbacks in _handlers[uid].values():
        for callbacks in prop_callbacks:
            for f, k, fn, bound_uid in callbacks:
                if fn is None:
                    continue
                try:
                    f.unbind_uid(k, bound_uid)
                except ReferenceError:
                    pass
    del _handlers[uid]