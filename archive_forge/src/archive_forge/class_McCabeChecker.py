from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
class McCabeChecker(object):
    """McCabe cyclomatic complexity checker."""
    name = 'mccabe'
    version = __version__
    _code = 'C901'
    _error_tmpl = 'C901 %r is too complex (%d)'
    max_complexity = -1

    def __init__(self, tree, filename):
        self.tree = tree

    @classmethod
    def add_options(cls, parser):
        flag = '--max-complexity'
        kwargs = {'default': -1, 'action': 'store', 'type': int, 'help': 'McCabe complexity threshold', 'parse_from_config': 'True'}
        config_opts = getattr(parser, 'config_options', None)
        if isinstance(config_opts, list):
            kwargs.pop('parse_from_config')
            parser.add_option(flag, **kwargs)
            parser.config_options.append('max-complexity')
        else:
            parser.add_option(flag, **kwargs)

    @classmethod
    def parse_options(cls, options):
        cls.max_complexity = int(options.max_complexity)

    def run(self):
        if self.max_complexity < 0:
            return
        visitor = PathGraphingAstVisitor()
        visitor.preorder(self.tree, visitor)
        for graph in visitor.graphs.values():
            if graph.complexity() > self.max_complexity:
                text = self._error_tmpl % (graph.entity, graph.complexity())
                yield (graph.lineno, graph.column, text, type(self))