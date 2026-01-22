from traitlets import (
from jupyter_server.extension.application import ExtensionApp
from notebook_shim import shim
class MockExtensionApp(shim.NotebookConfigShimMixin, ExtensionApp):
    """Mock an extension app that previously inherited NotebookApp."""
    name = 'mockextension'
    default_url = Unicode(config=True)
    enable_mathjax = Bool(config=True)
    allow_origin = Unicode(config=True)
    allow_origin_pat = Unicode(config=True)