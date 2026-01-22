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
def _from_dataset_id(processing_mode, service, dataset_id, element_spec, job_name=None, consumer_index=None, num_consumers=None, max_outstanding_requests=None, task_refresh_interval_hint_ms=None, data_transfer_protocol=None, cross_trainer_cache=None, target_workers='AUTO'):
    """Creates a dataset which reads data from the tf.data service.

  This transformation is similar to `from_dataset_id`, but supports additional
  parameters which we do not yet want to add to the public Python API.

  Args:
    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying
      how to shard the dataset among tf.data workers. See
      `tf.data.experimental.service.ShardingPolicy` for details. For backwards
      compatibility, `processing_mode` may also be set to the strings
      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
        address and `<protocol>` can optionally be used to override the default
        protocol to use. If it's a tuple, it should be (protocol, address).
    dataset_id: The id of the dataset to read from. This id is returned by
      `register_dataset` when the dataset is registered with the tf.data
      service.
    element_spec: A nested structure of `tf.TypeSpec`s representing the type of
      elements produced by the dataset. This argument is only required inside a
      tf.function. Use `tf.data.Dataset.element_spec` to get the element spec
      for a given dataset.
    job_name: (Optional.) The name of the job. If provided, it must be a
      non-empty string or tensor. This argument makes it possible for multiple
      datasets to share the same job. The default behavior is that the dataset
      creates anonymous, exclusively owned jobs.
    consumer_index: (Optional.) The index of the consumer in the range from `0`
      to `num_consumers`. Must be specified alongside `num_consumers`. When
      specified, consumers will read from the job in a strict round-robin order,
      instead of the default first-come-first-served order.
    num_consumers: (Optional.) The number of consumers which will consume from
      the job. Must be specified alongside `consumer_index`. When specified,
      consumers will read from the job in a strict round-robin order, instead of
      the default first-come-first-served order. When `num_consumers` is
      specified, the dataset must have infinite cardinality to prevent a
      producer from running out of data early and causing consumers to go out of
      sync.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    task_refresh_interval_hint_ms: (Optional.) A hint for how often to query the
      dispatcher for task changes.
    data_transfer_protocol: (Optional.) The protocol to use for transferring
      data with the tf.data service. By default, data is transferred using gRPC.
    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
      provided, dataset iteration will be shared across concurrently running
      trainers. See
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
      for details.
    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data
      runtime decides which workers to read from. If `"ANY"`, reads from any
      tf.data service workers. If `"LOCAL"`, only reads from local in-processs
      tf.data service workers. `"AUTO"` works well for most cases, while users
      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
      data copy if every TF worker colocates with a tf.data service worker.
      Consumers of a shared job must use the same `target_workers`. Defaults to
      `"AUTO"`.

  Returns:
    A `tf.data.Dataset` which reads from the tf.data service.
  """

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
    processing_mode = _get_validated_sharding_policy(processing_mode)
    if isinstance(service, tuple):
        protocol, address = service
    else:
        protocol, address = _parse_service(service)
    if job_name is not None:
        if not isinstance(job_name, str) and (not isinstance(job_name, tensor.Tensor)):
            raise ValueError(f'`job_name` must be a string or Tensor, but `job_name` was of type {type(job_name)}. job_name={job_name}.')
    if not element_spec:
        if not context.executing_eagerly():
            raise ValueError('In graph mode `element_spec` must be provided manually.')
        element_spec = _get_element_spec()
    dataset = _DataServiceDataset(dataset_id=dataset_id, processing_mode=processing_mode, address=address, element_spec=element_spec, protocol=protocol, data_transfer_protocol=data_transfer_protocol, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, cross_trainer_cache=cross_trainer_cache, target_workers=target_workers)
    if job_name is not None:
        options = options_lib.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
    return dataset