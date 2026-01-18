import pytest
import networkx as nx
def test_summary_graph(self):
    original_graph = self.build_original_graph()
    summary_graph = self.build_summary_graph()
    relationship_attributes = ('type',)
    generated_summary_graph = nx.snap_aggregation(original_graph, self.node_attributes)
    relabeled_summary_graph = self.deterministic_labels(generated_summary_graph)
    assert nx.is_isomorphic(summary_graph, relabeled_summary_graph)