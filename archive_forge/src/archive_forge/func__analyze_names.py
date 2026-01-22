import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _analyze_names(self, globals_or_nonlocals, type_):

    def raise_(message):
        self._add_syntax_error(base_name, message % (base_name.value, type_))
    params = []
    if self.node.type == 'funcdef':
        params = self.node.get_params()
    for base_name in globals_or_nonlocals:
        found_global_or_nonlocal = False
        for name in reversed(self._used_name_dict.get(base_name.value, [])):
            if name.start_pos > base_name.start_pos:
                found_global_or_nonlocal = True
            parent = name.parent
            if parent.type == 'param' and parent.name == name:
                continue
            if name.is_definition():
                if parent.type == 'expr_stmt' and parent.children[1].type == 'annassign':
                    if found_global_or_nonlocal:
                        base_name = name
                    raise_("annotated name '%s' can't be %s")
                    break
                else:
                    message = "name '%s' is assigned to before %s declaration"
            else:
                message = "name '%s' is used prior to %s declaration"
            if not found_global_or_nonlocal:
                raise_(message)
                break
        for param in params:
            if param.name.value == base_name.value:
                (raise_("name '%s' is parameter and %s"),)