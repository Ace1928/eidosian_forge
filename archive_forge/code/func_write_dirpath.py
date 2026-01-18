import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io
def write_dirpath(dirpath, strategy):
    """Returns the writing dir that should be used to save file distributedly.

  `dirpath` would be created if it doesn't exist.

  Args:
    dirpath: Original dirpath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.

  Returns:
    The writing dir path that should be used to save with distribution.
  """
    if strategy is None:
        strategy = distribute_lib.get_strategy()
    if strategy is None:
        return dirpath
    if not strategy.extended._in_multi_worker_mode():
        return dirpath
    if strategy.extended.should_checkpoint:
        return dirpath
    return _get_temp_dir(dirpath, strategy)