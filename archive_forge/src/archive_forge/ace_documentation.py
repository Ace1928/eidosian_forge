from bokeh.core.properties import (
from ..io.resources import bundled_files
from ..util import classproperty
from .enums import ace_themes
from .layout import HTMLBox

    A Bokeh model that wraps around a Ace editor and renders it inside
    a Bokeh plot.
    