import os
import re
from .._core import SHELL_NAMES, ShellDetectionFailure
from . import proc, ps
def _get_interpreter_shell(proc_name, proc_args):
    """Get shell invoked via an interpreter.

    Some shells are implemented on, and invoked with an interpreter, e.g. xonsh
    is commonly executed with an executable Python script. This detects what
    script the interpreter is actually running, and check whether that looks
    like a shell.

    See sarugaku/shellingham#26 for rational.
    """
    for pattern, shell_names in _INTERPRETER_SHELL_NAMES:
        if not pattern.match(proc_name):
            continue
        for arg in proc_args:
            name = os.path.basename(arg).lower()
            if os.path.isfile(arg) and name in shell_names:
                return (name, arg)
    return None