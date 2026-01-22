import math
import numbers
import os
import re
import sys
import time
import types
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.test.ParameterizedBenchmark', v1=[])
class ParameterizedBenchmark(_BenchmarkRegistrar):
    """Metaclass to generate parameterized benchmarks.

  Use this class as a metaclass and override the `_benchmark_parameters` to
  generate multiple benchmark test cases. For example:

  class FooBenchmark(metaclass=tf.test.ParameterizedBenchmark,
                     tf.test.Benchmark):
    # The `_benchmark_parameters` is expected to be a list with test cases.
    # Each of the test case is a tuple, with the first time to be test case
    # name, followed by any number of the parameters needed for the test case.
    _benchmark_parameters = [
      ('case_1', Foo, 1, 'one'),
      ('case_2', Bar, 2, 'two'),
    ]

    def benchmark_test(self, target_class, int_param, string_param):
      # benchmark test body

  The example above will generate two benchmark test cases:
  "benchmark_test__case_1" and "benchmark_test__case_2".
  """

    def __new__(mcs, clsname, base, attrs):
        param_config_list = attrs['_benchmark_parameters']

        def create_benchmark_function(original_benchmark, params):
            return lambda self: original_benchmark(self, *params)
        for name in attrs.copy().keys():
            if not name.startswith('benchmark'):
                continue
            original_benchmark = attrs[name]
            del attrs[name]
            for param_config in param_config_list:
                test_name_suffix = param_config[0]
                params = param_config[1:]
                benchmark_name = name + '__' + test_name_suffix
                if benchmark_name in attrs:
                    raise Exception('Benchmark named {} already defined.'.format(benchmark_name))
                benchmark = create_benchmark_function(original_benchmark, params)
                attrs[benchmark_name] = _rename_function(benchmark, 1, benchmark_name)
        return super().__new__(mcs, clsname, base, attrs)