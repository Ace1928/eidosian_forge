import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='error_node')
class _ExpectIndentedBlock(IndentationRule):
    message = 'expected an indented block'

    def get_node(self, node):
        leaf = node.get_next_leaf()
        return list(leaf._split_prefix())[-1]

    def is_issue(self, node):
        return node.children[-1].type == 'newline'