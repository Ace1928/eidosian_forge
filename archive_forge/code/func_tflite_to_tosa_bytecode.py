from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export
@tf_export('mlir.experimental.tflite_to_tosa_bytecode')
def tflite_to_tosa_bytecode(flatbuffer, bytecode, use_external_constant=False, ordered_input_arrays=None, ordered_output_arrays=None):
    """Converts TFLite flatbuffer to TOSA dialect in MLIR bytecode.

  Args:
    flatbuffer: Path to flatbuffer.
    bytecode: Path to output bytecode.
    use_external_constant: Whether to create `tfl.external_const` instead of
      `tfl.const`.
    ordered_input_arrays:
    ordered_output_arrays: If ordered_output_arrays is not empty, then the
      function will only return nodes in ordered_output_arrays in the same order
  """
    pywrap_mlir.experimental_tflite_to_tosa_bytecode(flatbuffer, bytecode, use_external_constant, ordered_input_arrays, ordered_output_arrays)