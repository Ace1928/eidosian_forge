from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _is_valid_classification_signature(signature_def):
    """Determine whether the argument is a servable 'classify' SignatureDef."""
    if signature_def.method_name != signature_constants.CLASSIFY_METHOD_NAME:
        return False
    if set(signature_def.inputs.keys()) != set([signature_constants.CLASSIFY_INPUTS]):
        return False
    if signature_def.inputs[signature_constants.CLASSIFY_INPUTS].dtype != types_pb2.DT_STRING:
        return False
    allowed_outputs = set([signature_constants.CLASSIFY_OUTPUT_CLASSES, signature_constants.CLASSIFY_OUTPUT_SCORES])
    if not signature_def.outputs.keys():
        return False
    if set(signature_def.outputs.keys()) - allowed_outputs:
        return False
    if signature_constants.CLASSIFY_OUTPUT_CLASSES in signature_def.outputs and signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES].dtype != types_pb2.DT_STRING:
        return False
    if signature_constants.CLASSIFY_OUTPUT_SCORES in signature_def.outputs and signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES].dtype != types_pb2.DT_FLOAT:
        return False
    return True