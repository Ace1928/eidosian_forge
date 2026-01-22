from functools import wraps
from copy import deepcopy
from traitlets import TraitError
from traitlets.config.loader import (
from jupyter_core.application import JupyterApp
from jupyter_server.serverapp import ServerApp
from jupyter_server.extension.application import ExtensionApp
from .traits import NotebookAppTraits
def EXTAPP_TO_SVAPP_SHIM_MSG(trait_name, extapp_name):
    return "'{trait_name}' has moved from {extapp_name} to ServerApp. Be sure to update your config before our next release.".format(trait_name=trait_name, extapp_name=extapp_name)