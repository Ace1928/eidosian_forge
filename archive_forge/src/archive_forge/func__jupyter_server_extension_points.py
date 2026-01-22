from traitlets import (
from jupyter_server.extension.application import ExtensionApp
from notebook_shim import shim
def _jupyter_server_extension_points():
    return [{'module': 'notebook_shim.tests.mockextension', 'app': MockExtensionApp}]