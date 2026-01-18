from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def uncompress(element, output_spec):
    """Uncompress a compressed dataset element.

  Args:
    element: A scalar variant tensor to uncompress. The element should have been
      created by calling `compress`.
    output_spec: A nested structure of `tf.TypeSpec` representing the type(s) of
      the uncompressed element.

  Returns:
    The uncompressed element.
  """
    flat_types = structure.get_flat_tensor_types(output_spec)
    flat_shapes = structure.get_flat_tensor_shapes(output_spec)
    tensor_list = ged_ops.uncompress_element(element, output_types=flat_types, output_shapes=flat_shapes)
    return structure.from_tensor_list(output_spec, tensor_list)