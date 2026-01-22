import os
import re
from pathlib import Path
from importlib.machinery import all_suffixes
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ContextualizedNode
from jedi.inference.helpers import is_string, get_str_or_none
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FileIO
from jedi import settings
from jedi import debug
def _paths_from_list_modifications(module_context, trailer1, trailer2):
    """ extract the path from either "sys.path.append" or "sys.path.insert" """
    if not (trailer1.type == 'trailer' and trailer1.children[0] == '.' and (trailer2.type == 'trailer') and (trailer2.children[0] == '(') and (len(trailer2.children) == 3)):
        return
    name = trailer1.children[1].value
    if name not in ['insert', 'append']:
        return
    arg = trailer2.children[1]
    if name == 'insert' and len(arg.children) in (3, 4):
        arg = arg.children[2]
    for value in module_context.create_context(arg).infer_node(arg):
        p = get_str_or_none(value)
        if p is None:
            continue
        abs_path = _abs_path(module_context, p)
        if abs_path is not None:
            yield abs_path