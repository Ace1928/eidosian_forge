import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
class FunctionVisitor(transformer.Base):
    """AST visitor that applies type inference to each function separately."""

    def __init__(self, source_info, graphs, resolver):
        super(FunctionVisitor, self).__init__(source_info)
        self.graphs = graphs
        self.resolver = resolver

    def visit_FunctionDef(self, node):
        subgraph = self.graphs[node]
        scope = anno.getanno(node, annos.NodeAnno.ARGS_AND_BODY_SCOPE)
        closure_types = anno.getanno(node, anno.Static.CLOSURE_TYPES, {})
        analyzer = Analyzer(subgraph, self.resolver, self.ctx.info.namespace, scope, closure_types)
        analyzer.visit_forward()
        node.body = self.visit_block(node.body)
        return node