import os
from traitlets import Unicode
from ipywidgets import DOMWidget
from ._version import semver
class VueComponent(DOMWidget):
    _model_name = Unicode('VueComponentModel').tag(sync=True)
    _model_module = Unicode('jupyter-vue').tag(sync=True)
    _model_module_version = Unicode(semver).tag(sync=True)
    name = Unicode().tag(sync=True)
    component = Unicode().tag(sync=True)