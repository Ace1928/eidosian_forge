import os
import sys
from typing import Any, MutableMapping, Tuple
import click
from ._completion_classes import completion_init
from ._completion_shared import Shells, get_completion_script, install
from .models import ParamMeta
from .params import Option
from .utils import get_params_from_function
def _install_completion_placeholder_function(install_completion: bool=Option(None, '--install-completion', is_flag=True, callback=install_callback, expose_value=False, help='Install completion for the current shell.'), show_completion: bool=Option(None, '--show-completion', is_flag=True, callback=show_callback, expose_value=False, help='Show completion for the current shell, to copy it or customize the installation.')) -> Any:
    pass