import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output
def _determine_editor(self, options: Values) -> str:
    if options.editor is not None:
        return options.editor
    elif 'VISUAL' in os.environ:
        return os.environ['VISUAL']
    elif 'EDITOR' in os.environ:
        return os.environ['EDITOR']
    else:
        raise PipError('Could not determine editor to use.')