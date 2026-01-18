from tensorflow.core.protobuf import config_pb2
from tensorflow.lite.python import interpreter
from tensorflow.lite.python import lite
from tensorflow.python.eager import def_function
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver
def tflite_convert(fn, input_templates):
    """Converts the provided fn to tf.lite model.

  Args:
    fn: A callable that expects a list of inputs like input_templates that
      returns a tensor or structure of tensors.
    input_templates: A list of Tensors, ndarrays or TensorSpecs describing the
      inputs that fn expects. The actual values of the Tensors or ndarrays are
      unused.

  Returns:
    The serialized tf.lite model.
  """
    fn = def_function.function(fn)
    concrete_func = fn.get_concrete_function(*input_templates)
    converter = lite.TFLiteConverterV2([concrete_func])
    return converter.convert()