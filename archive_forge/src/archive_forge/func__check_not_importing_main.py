import os
import sys
import runpy
import types
from . import get_start_method, set_start_method
from . import process
from .context import reduction
from . import util
def _check_not_importing_main():
    if getattr(process.current_process(), '_inheriting', False):
        raise RuntimeError('\n        An attempt has been made to start a new process before the\n        current process has finished its bootstrapping phase.\n\n        This probably means that you are not using fork to start your\n        child processes and you have forgotten to use the proper idiom\n        in the main module:\n\n            if __name__ == \'__main__\':\n                freeze_support()\n                ...\n\n        The "freeze_support()" line can be omitted if the program\n        is not going to be frozen to produce an executable.')