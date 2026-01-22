import collections
import threading
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.tf_export import tf_export
@tf_export('profiler.experimental.ProfilerOptions', v1=[])
class ProfilerOptions(collections.namedtuple('ProfilerOptions', ['host_tracer_level', 'python_tracer_level', 'device_tracer_level', 'delay_ms'])):
    """Options for finer control over the profiler.

  Use `tf.profiler.experimental.ProfilerOptions` to control `tf.profiler`
  behavior.

  Fields:
    host_tracer_level: Adjust CPU tracing level. Values are: `1` - critical info
      only, `2` - info, `3` - verbose. [default value is `2`]
    python_tracer_level: Toggle tracing of Python function calls. Values are:
      `1` - enabled, `0` - disabled [default value is `0`]
    device_tracer_level: Adjust device (TPU/GPU) tracing level. Values are:
      `1` - enabled, `0` - disabled [default value is `1`]
    delay_ms: Requests for all hosts to start profiling at a timestamp that is
      `delay_ms` away from the current time. `delay_ms` is in milliseconds. If
      zero, each host will start profiling immediately upon receiving the
      request. Default value is `None`, allowing the profiler guess the best
      value.
  """

    def __new__(cls, host_tracer_level=2, python_tracer_level=0, device_tracer_level=1, delay_ms=None):
        return super(ProfilerOptions, cls).__new__(cls, host_tracer_level, python_tracer_level, device_tracer_level, delay_ms)