import inspect
from pathlib import Path
from jedi.parser_utils import get_cached_code_lines
from jedi import settings
from jedi.cache import memoize_method
from jedi.inference import compiled
from jedi.file_io import FileIO
from jedi.inference.names import NameWrapper
from jedi.inference.base_value import ValueSet, ValueWrapper, NO_VALUES
from jedi.inference.value import ModuleValue
from jedi.inference.cache import inference_state_function_cache, \
from jedi.inference.compiled.access import ALLOWED_GETITEM_TYPES, get_api_type
from jedi.inference.gradual.conversion import to_stub
from jedi.inference.context import CompiledContext, CompiledModuleContext, \
def _find_syntax_node_name(inference_state, python_object):
    original_object = python_object
    try:
        python_object = _get_object_to_check(python_object)
        path = inspect.getsourcefile(python_object)
    except (OSError, TypeError):
        return None
    path = None if path is None else Path(path)
    try:
        if path is None or not path.exists():
            return None
    except OSError:
        return None
    file_io = FileIO(path)
    module_node = _load_module(inference_state, path)
    if inspect.ismodule(python_object):
        code_lines = get_cached_code_lines(inference_state.grammar, path)
        return (module_node, module_node, file_io, code_lines)
    try:
        name_str = python_object.__name__
    except AttributeError:
        return None
    if name_str == '<lambda>':
        return None
    names = module_node.get_used_names().get(name_str, [])
    names = [n for n in names if n.parent.type in ('funcdef', 'classdef') and n.parent.name == n]
    if not names:
        return None
    try:
        code = python_object.__code__
        line_nr = code.co_firstlineno
    except AttributeError:
        pass
    else:
        line_names = [name for name in names if name.start_pos[0] == line_nr]
        if line_names:
            names = line_names
    code_lines = get_cached_code_lines(inference_state.grammar, path)
    tree_node = names[-1].parent
    if tree_node.type == 'funcdef' and get_api_type(original_object) == 'instance':
        return None
    return (module_node, tree_node, file_io, code_lines)