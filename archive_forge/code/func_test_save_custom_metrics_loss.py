import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_save_custom_metrics_loss(tmp_path):

    def custom_metric(y_pred, y_true):
        return 1

    def custom_loss(y_pred, y_true):
        return y_pred - y_true
    head = ak.ClassificationHead(loss=custom_loss, metrics=['accuracy', custom_metric])
    input_node = ak.Input()
    output_node = head(input_node)
    graph = graph_module.Graph(input_node, output_node)
    path = os.path.join(tmp_path, 'graph')
    graph.save(path)
    new_graph = graph_module.load_graph(path, custom_objects={'custom_metric': custom_metric, 'custom_loss': custom_loss})
    assert new_graph.blocks[0].metrics[1](0, 0) == 1
    assert new_graph.blocks[0].loss(3, 2) == 1