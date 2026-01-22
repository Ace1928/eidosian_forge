import sys
from typing import List
from pathlib import Path
from parso.tree import search_ancestor
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.imports import goto_import, load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.helpers import infer_call_of_leaf
def _is_a_pytest_param_and_inherited(param_name):
    """
    Pytest params are either in a `test_*` function or have a pytest fixture
    with the decorator @pytest.fixture.

    This is a heuristic and will work in most cases.
    """
    funcdef = search_ancestor(param_name.tree_name, 'funcdef')
    if funcdef is None:
        return (False, False)
    decorators = funcdef.get_decorators()
    return (_is_pytest_func(funcdef.name.value, decorators), funcdef.name.value == param_name.string_name)