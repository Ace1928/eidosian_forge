from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export
Converts TFLite flatbuffer to TOSA dialect in MLIR bytecode.

  Args:
    flatbuffer: Path to flatbuffer.
    bytecode: Path to output bytecode.
    use_external_constant: Whether to create `tfl.external_const` instead of
      `tfl.const`.
    ordered_input_arrays:
    ordered_output_arrays: If ordered_output_arrays is not empty, then the
      function will only return nodes in ordered_output_arrays in the same order
  