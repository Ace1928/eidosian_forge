from typing import Set, Tuple
from onnx import ModelProto
from onnxruntime.transformers.onnx_model import OnnxModel
from .. import PreprocessorPass
class ExcludeNodeAfter(PreprocessorPass):

    def __init__(self, parent_operator_type: str, operator_type_to_exclude: str):
        super().__init__()
        self.parent_operator_type = parent_operator_type
        self.operator_type_to_exclude = operator_type_to_exclude

    def __call__(self, graph: ModelProto, model: OnnxModel) -> Tuple[Set[str], Set[str]]:
        candidate_nodes_to_exclude = {candidate_input: candidate.name for candidate in model.get_nodes_by_op_type(self.operator_type_to_exclude) for candidate_input in candidate.input}
        parent_node = {node_output: node.name for node in model.get_nodes_by_op_type(self.parent_operator_type) for node_output in node.output}
        to_exclude = set(candidate_nodes_to_exclude.keys()).intersection(parent_node.keys())
        nodes_to_exclude = {candidate_nodes_to_exclude[node] for node in to_exclude}
        return (set(), nodes_to_exclude)