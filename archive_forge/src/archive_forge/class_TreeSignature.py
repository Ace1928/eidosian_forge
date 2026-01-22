from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
class TreeSignature(AbstractSignature):

    def __init__(self, value, function_value=None, is_bound=False):
        super().__init__(value, is_bound)
        self._function_value = function_value or value

    def bind(self, value):
        return TreeSignature(value, self._function_value, is_bound=True)

    @property
    def _annotation(self):
        if self.value.is_class():
            return None
        return self._function_value.tree_node.annotation

    @property
    def annotation_string(self):
        a = self._annotation
        if a is None:
            return ''
        return a.get_code(include_prefix=False)

    @memoize_method
    def get_param_names(self, resolve_stars=False):
        params = self._function_value.get_param_names()
        if resolve_stars:
            from jedi.inference.star_args import process_params
            params = process_params(params)
        if self.is_bound:
            return params[1:]
        return params

    def matches_signature(self, arguments):
        from jedi.inference.param import get_executed_param_names_and_issues
        executed_param_names, issues = get_executed_param_names_and_issues(self._function_value, arguments)
        if issues:
            return False
        matches = all((executed_param_name.matches_signature() for executed_param_name in executed_param_names))
        if debug.enable_notice:
            tree_node = self._function_value.tree_node
            signature = parser_utils.get_signature(tree_node)
            if matches:
                debug.dbg('Overloading match: %s@%s (%s)', signature, tree_node.start_pos[0], arguments, color='BLUE')
            else:
                debug.dbg('Overloading no match: %s@%s (%s)', signature, tree_node.start_pos[0], arguments, color='BLUE')
        return matches