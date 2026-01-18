import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
Which profile step to use for profiling.

    The 'step' here refers to the step defined by `Profiler.add_step()` API.

    Args:
      step: When multiple steps of profiles are available, select which step's
         profile to use. If -1, use average of all available steps.
    Returns:
      self
    