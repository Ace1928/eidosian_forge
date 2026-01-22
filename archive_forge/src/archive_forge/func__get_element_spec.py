import enum
import functools
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _get_element_spec():
    """Fetches the element spec from the server."""
    data_service_metadata = None
    dataset_id_val = tensor_util.constant_value(dataset_id)
    try:
        data_service_metadata = _pywrap_server_lib.TF_DATA_GetDataServiceMetadataByID(dataset_id_val, address, protocol)
    except NotImplementedError as err:
        raise ValueError('The tf.data service is running an earlier version of TensorFlow that requires specifying `element_spec` as an argument to `from_dataset_id`. Please either supply an element spec or update the tf.data service to the latest version.') from err
    except RuntimeError:
        pass
    if not data_service_metadata or not data_service_metadata.element_spec:
        dataset_id_val = tensor_util.constant_value(dataset_id)
        raise ValueError(f'Failed to fetch element spec for dataset id {dataset_id_val} from tf.data service. If the dataset was registered in graph mode or inside a tf.function, the `element_spec` must be specified as an argument to `from_dataset_id`.')
    struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
    struct_pb.ParseFromString(data_service_metadata.element_spec)
    return nested_structure_coder.decode_proto(struct_pb)