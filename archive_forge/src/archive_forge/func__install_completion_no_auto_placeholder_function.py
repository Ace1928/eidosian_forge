import os
import sys
from typing import Any, MutableMapping, Tuple
import click
from ._completion_classes import completion_init
from ._completion_shared import Shells, get_completion_script, install
from .models import ParamMeta
from .params import Option
from .utils import get_params_from_function
def _install_completion_no_auto_placeholder_function(install_completion: Shells=Option(None, callback=install_callback, expose_value=False, help='Install completion for the specified shell.'), show_completion: Shells=Option(None, callback=show_callback, expose_value=False, help='Show completion for the specified shell, to copy it or customize the installation.')) -> Any:
    pass