from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
class ParseTreeBuilder:

    def __init__(self, rules, tree_class, propagate_positions=False, ambiguous=False, maybe_placeholders=False):
        self.tree_class = tree_class
        self.propagate_positions = propagate_positions
        self.ambiguous = ambiguous
        self.maybe_placeholders = maybe_placeholders
        self.rule_builders = list(self._init_builders(rules))

    def _init_builders(self, rules):
        propagate_positions = make_propagate_positions(self.propagate_positions)
        for rule in rules:
            options = rule.options
            keep_all_tokens = options.keep_all_tokens
            expand_single_child = options.expand1
            wrapper_chain = list(filter(None, [(expand_single_child and (not rule.alias)) and ExpandSingleChild, maybe_create_child_filter(rule.expansion, keep_all_tokens, self.ambiguous, options.empty_indices if self.maybe_placeholders else None), propagate_positions, self.ambiguous and maybe_create_ambiguous_expander(self.tree_class, rule.expansion, keep_all_tokens), self.ambiguous and partial(AmbiguousIntermediateExpander, self.tree_class)]))
            yield (rule, wrapper_chain)

    def create_callback(self, transformer=None):
        callbacks = {}
        default_handler = getattr(transformer, '__default__', None)
        if default_handler:

            def default_callback(data, children):
                return default_handler(data, children, None)
        else:
            default_callback = self.tree_class
        for rule, wrapper_chain in self.rule_builders:
            user_callback_name = rule.alias or rule.options.template_source or rule.origin.name
            try:
                f = getattr(transformer, user_callback_name)
                wrapper = getattr(f, 'visit_wrapper', None)
                if wrapper is not None:
                    f = apply_visit_wrapper(f, user_callback_name, wrapper)
                elif isinstance(transformer, Transformer_InPlace):
                    f = inplace_transformer(f)
            except AttributeError:
                f = partial(default_callback, user_callback_name)
            for w in wrapper_chain:
                f = w(f)
            if rule in callbacks:
                raise GrammarError("Rule '%s' already exists" % (rule,))
            callbacks[rule] = f
        return callbacks