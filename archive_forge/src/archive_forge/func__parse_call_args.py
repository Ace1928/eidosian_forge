import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _parse_call_args(call: ast.Call):
    """Parse arguments of a Signal call/Slot decorator (type list)."""
    result: Arguments = []
    for n, arg in enumerate(call.args):
        par_name = f'a{n + 1}'
        par_type = _python_to_cpp_type(_name(arg))
        result.append({'name': par_name, 'type': par_type})
    return result