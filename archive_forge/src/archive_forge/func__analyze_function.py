import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos
def _analyze_function(self, node, is_lambda):
    parent_analyzer = self.current_analyzer
    analyzer = Analyzer(self.graphs[node], self.include_annotations)
    analyzer.visit_reverse()
    self.current_analyzer = analyzer
    node = self.generic_visit(node)
    self.current_analyzer = parent_analyzer
    return node