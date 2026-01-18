from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import load
from onnx.defs import onnx_opset_version
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from onnx.model_container import ModelContainer
from onnx.onnx_pb import (
from onnx.reference.op_run import (
from onnx.reference.ops_optimized import optimized_operators
Executes the onnx model.

        Args:
            output_names: requested outputs by names, None for all
            feed_inputs: dictionary `{ input name: input value }`
            attributes: attributes value if the instance runs a
                FunctionProto

        Returns:
            list of requested outputs
        