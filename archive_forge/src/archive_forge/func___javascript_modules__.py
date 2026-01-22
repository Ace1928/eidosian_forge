from bokeh.core.properties import (
from bokeh.events import ModelEvent
from bokeh.models import ColumnDataSource
from ..config import config
from ..io.resources import JS_VERSION, bundled_files
from ..util import classproperty
from .layout import HTMLBox
@classproperty
def __javascript_modules__(cls):
    return [js for js in bundled_files(cls, 'javascript_modules') if 'wasm' not in js and 'worker' not in js]