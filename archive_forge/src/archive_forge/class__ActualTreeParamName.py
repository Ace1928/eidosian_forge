from abc import abstractmethod
from inspect import Parameter
from typing import Optional, Tuple
from parso.tree import search_ancestor
from jedi.parser_utils import find_statement_documentation, clean_scope_docstring
from jedi.inference.utils import unite
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.cache import inference_state_method_cache
from jedi.inference import docstrings
from jedi.cache import memoize_method
from jedi.inference.helpers import deep_ast_copy, infer_call_of_leaf
from jedi.plugins import plugin_manager
class _ActualTreeParamName(BaseTreeParamName):

    def __init__(self, function_value, tree_name):
        super().__init__(function_value.get_default_param_context(), tree_name)
        self.function_value = function_value

    def _get_param_node(self):
        return search_ancestor(self.tree_name, 'param')

    @property
    def annotation_node(self):
        return self._get_param_node().annotation

    def infer_annotation(self, execute_annotation=True, ignore_stars=False):
        from jedi.inference.gradual.annotation import infer_param
        values = infer_param(self.function_value, self._get_param_node(), ignore_stars=ignore_stars)
        if execute_annotation:
            values = values.execute_annotation()
        return values

    def infer_default(self):
        node = self.default_node
        if node is None:
            return NO_VALUES
        return self.parent_context.infer_node(node)

    @property
    def default_node(self):
        return self._get_param_node().default

    def get_kind(self):
        tree_param = self._get_param_node()
        if tree_param.star_count == 1:
            return Parameter.VAR_POSITIONAL
        if tree_param.star_count == 2:
            return Parameter.VAR_KEYWORD
        if tree_param.name.value.startswith('__'):
            return Parameter.POSITIONAL_ONLY
        parent = tree_param.parent
        param_appeared = False
        for p in parent.children:
            if param_appeared:
                if p == '/':
                    return Parameter.POSITIONAL_ONLY
            else:
                if p == '*':
                    return Parameter.KEYWORD_ONLY
                if p.type == 'param':
                    if p.star_count:
                        return Parameter.KEYWORD_ONLY
                    if p == tree_param:
                        param_appeared = True
        return Parameter.POSITIONAL_OR_KEYWORD

    def infer(self):
        values = self.infer_annotation()
        if values:
            return values
        doc_params = docstrings.infer_param(self.function_value, self._get_param_node())
        return doc_params