from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
def _init_builders(self, rules):
    propagate_positions = make_propagate_positions(self.propagate_positions)
    for rule in rules:
        options = rule.options
        keep_all_tokens = options.keep_all_tokens
        expand_single_child = options.expand1
        wrapper_chain = list(filter(None, [(expand_single_child and (not rule.alias)) and ExpandSingleChild, maybe_create_child_filter(rule.expansion, keep_all_tokens, self.ambiguous, options.empty_indices if self.maybe_placeholders else None), propagate_positions, self.ambiguous and maybe_create_ambiguous_expander(self.tree_class, rule.expansion, keep_all_tokens), self.ambiguous and partial(AmbiguousIntermediateExpander, self.tree_class)]))
        yield (rule, wrapper_chain)