import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _global_registry_option(name, help, registry=None, **kwargs):
    Option.OPTIONS[name] = RegistryOption(name, help, registry, **kwargs)