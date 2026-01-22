import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization
@staticmethod
def _get_static_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)