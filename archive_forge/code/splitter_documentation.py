from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
If True, will automatically change size to take up the same
    proportion of the parent widget when it is resized, while
    staying within :attr:`min_size` and :attr:`max_size`. As long as
    these attributes can be satisfied, this stops the
    :class:`Splitter` from exceeding the parent size during rescaling.

    :attr:`rescale_with_parent` is a
    :class:`~kivy.properties.BooleanProperty` and defaults to False.

    .. versionadded:: 1.9.0
    