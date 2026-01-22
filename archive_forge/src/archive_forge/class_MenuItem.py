import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
class MenuItem:
    """A class for an item in a text-based menu."""

    def __init__(self, caption, content, enabled=True):
        """Menu constructor.

    TODO(cais): Nested menu is currently not supported. Support it.

    Args:
      caption: (str) caption of the menu item.
      content: Content of the menu item. For a menu item that triggers
        a command, for example, content is the command string.
      enabled: (bool) whether this menu item is enabled.
    """
        self._caption = caption
        self._content = content
        self._enabled = enabled

    @property
    def caption(self):
        return self._caption

    @property
    def type(self):
        return self._node_type

    @property
    def content(self):
        return self._content

    def is_enabled(self):
        return self._enabled

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True