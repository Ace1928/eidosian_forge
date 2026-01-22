from tensorflow.python.framework.python_memory_checker import _PythonMemoryChecker
from tensorflow.python.profiler import trace
from tensorflow.python.util import tf_inspect
def _get_test_name_best_effort():
    """If available, return the current test name. Otherwise, `None`."""
    for stack in tf_inspect.stack():
        function_name = stack[3]
        if function_name.startswith('test'):
            try:
                class_name = stack[0].f_locals['self'].__class__.__name__
                return class_name + '.' + function_name
            except:
                pass
    return None