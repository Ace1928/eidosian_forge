import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(value='yield')
@ErrorFinder.register_rule(value='return')
class _ReturnAndYieldChecks(SyntaxRule):
    message = "'return' with value in async generator"
    message_async_yield = "'yield' inside async function"

    def get_node(self, leaf):
        return leaf.parent

    def is_issue(self, leaf):
        if self._normalizer.context.node.type != 'funcdef':
            self.add_issue(self.get_node(leaf), message="'%s' outside function" % leaf.value)
        elif self._normalizer.context.is_async_funcdef() and any(self._normalizer.context.node.iter_yield_exprs()):
            if leaf.value == 'return' and leaf.parent.type == 'return_stmt':
                return True