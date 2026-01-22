import collections
import enum
import functools
from typing import Text
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics
def append_error(self, error_data: converter_error_data_pb2.ConverterErrorData):
    self.errors.append(error_data)