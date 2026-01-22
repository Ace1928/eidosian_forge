import sys
from typing import List
from pathlib import Path
from parso.tree import search_ancestor
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.imports import goto_import, load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.helpers import infer_call_of_leaf
def _goto_pytest_fixture(module_context, name, skip_own_module):
    for module_context in _iter_pytest_modules(module_context, skip_own_module=skip_own_module):
        names = FixtureFilter(module_context).get(name)
        if names:
            return names