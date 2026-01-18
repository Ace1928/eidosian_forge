from functools import partial
from kivy.clock import Clock
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty, OptionProperty, \
Switch to a specific panel header.

        .. versionchanged:: 1.10.0

        If used with `do_scroll=True`, it scrolls
        to the header's tab too.

        :meth:`switch_to` cannot be called from within the
        :class:`TabbedPanel` or its subclass' ``__init__`` method.
        If that is required, use the ``Clock`` to schedule it. See `discussion
        <https://github.com/kivy/kivy/issues/3493#issuecomment-121567969>`_
        for full example.
        