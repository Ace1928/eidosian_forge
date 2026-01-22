import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
def _proces_function(self, node):
    parent_analyzer = self.current_analyzer
    subgraph = self.graphs[node]
    if self.current_analyzer is not None and node in self.current_analyzer.graph.index:
        cfg_node = self.current_analyzer.graph.index[node]
        defined_in = self.current_analyzer.in_[cfg_node].value
    else:
        defined_in = ()
    analyzer = Analyzer(subgraph, defined_in)
    analyzer.visit_forward()
    self.current_analyzer = analyzer
    node = self.generic_visit(node)
    self.current_analyzer = parent_analyzer
    return node