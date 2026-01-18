from tensorflow.python.framework.python_memory_checker import _PythonMemoryChecker
from tensorflow.python.profiler import trace
from tensorflow.python.util import tf_inspect
Raises an exception if there are new Python objects created.

    It computes the number of new Python objects per type using the first and
    the last snapshots.

    Args:
      threshold: A dictionary of [Type name string], [count] pair. It won't
        raise an exception if the new Python objects are under this threshold.
    