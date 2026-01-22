from typing import Set, Tuple
from onnx import ModelProto
from onnxruntime.transformers.onnx_model import OnnxModel
from .. import PreprocessorPass
class ExcludeNodeFollowedBy(PreprocessorPass):

    def __init__(self, operator_type_to_exclude: str, following_operator_type: str):
        super().__init__()
        self.operator_type_to_exclude = operator_type_to_exclude
        self.following_operator_type = following_operator_type

    def __call__(self, _: ModelProto, model: OnnxModel) -> Tuple[Set[str], Set[str]]:
        candidate_nodes_to_exclude = {candidate_output: candidate.name for candidate in model.get_nodes_by_op_type(self.operator_type_to_exclude) for candidate_output in candidate.output}
        nodes_of_following_type = {node_input: node.name for node in model.get_nodes_by_op_type(self.following_operator_type) for node_input in node.input}
        to_exclude = set(candidate_nodes_to_exclude.keys()).intersection(nodes_of_following_type.keys())
        nodes_to_exclude = {candidate_nodes_to_exclude[node] for node in to_exclude}
        return (set(), nodes_to_exclude)