from __future__ import absolute_import
import ipywidgets as widgets
from bokeh.models import CustomJS
from traitlets import Unicode
import ipyvolume
def _ensure_widget_manager_hack():
    global wmh
    if not wmh:
        wmh = WidgetManagerHackModel()