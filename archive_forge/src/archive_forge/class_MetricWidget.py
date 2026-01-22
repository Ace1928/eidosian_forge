import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization
class MetricWidget(DOMWidget):
    _view_name = Unicode('CatboostWidgetView').tag(sync=True)
    _model_name = Unicode('CatboostWidgetModel').tag(sync=True)
    _view_module = Unicode('catboost-widget').tag(sync=True)
    _model_module = Unicode('catboost-widget').tag(sync=True)
    _view_module_version = Unicode('^1.0.0').tag(sync=True)
    _model_module_version = Unicode('^1.0.0').tag(sync=True)
    data = Dict({}).tag(sync=True, **widget_serialization)

    @default('layout')
    def _default_layout(self):
        return Layout(height='500px', align_self='stretch')