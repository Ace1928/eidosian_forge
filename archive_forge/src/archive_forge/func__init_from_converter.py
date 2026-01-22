import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _init_from_converter(self, options: QuantizationDebugOptions, converter: TFLiteConverter, calibrated_model: Optional[bytes]=None, float_model: Optional[bytes]=None) -> None:
    """Convert the model and apply options.

    Converts the quantized model and initializes a quantized model interpreter
    with the quantized model. Returns a float model interpreter if float model
    is provided.

    Args:
      options: a QuantizationDebugOptions object.
      converter: an initialized tf.lite.TFLiteConverter.
      calibrated_model: Calibrated model bytes.
      float_model: Float model bytes.
    """
    self.quant_model = convert.mlir_quantize(calibrated_model, disable_per_channel=converter._experimental_disable_per_channel, fully_quantize=options.fully_quantize, enable_numeric_verify=True, denylisted_ops=options.denylisted_ops, denylisted_nodes=options.denylisted_nodes)
    self._quant_interpreter = _interpreter.Interpreter(model_content=self.quant_model)
    self._float_interpreter = None
    if float_model is not None:
        self._float_interpreter = _interpreter.Interpreter(model_content=float_model)