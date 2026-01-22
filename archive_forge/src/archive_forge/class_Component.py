import collections
import enum
import functools
from typing import Text
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics
class Component(enum.Enum):
    """Enum class defining name of the converter components."""
    PREPARE_TF_MODEL = 'PREPARE_TF_MODEL'
    CONVERT_TF_TO_TFLITE_MODEL = 'CONVERT_TF_TO_TFLITE_MODEL'
    OPTIMIZE_TFLITE_MODEL = 'OPTIMIZE_TFLITE_MODEL'