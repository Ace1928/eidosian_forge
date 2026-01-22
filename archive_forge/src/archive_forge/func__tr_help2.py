import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def _tr_help2(content):
    """Translate lines escaped with: ??

    A naked help line should fire the intro help screen (shell.show_usage())
    """
    if not content:
        return 'get_ipython().show_usage()'
    return _make_help_call(content, '??')