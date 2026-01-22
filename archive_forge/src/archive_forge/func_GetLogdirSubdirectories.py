import collections
import os
import re
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def GetLogdirSubdirectories(path):
    """Obtains all subdirectories with events files.

    The order of the subdirectories returned is unspecified. The internal logic
    that determines order varies by scenario.

    Args:
      path: The path to a directory under which to find subdirectories.

    Returns:
      A tuple of absolute paths of all subdirectories each with at least 1 events
      file directly within the subdirectory.

    Raises:
      ValueError: If the path passed to the method exists and is not a directory.
    """
    if not tf.io.gfile.exists(path):
        return ()
    if not tf.io.gfile.isdir(path):
        raise ValueError('GetLogdirSubdirectories: path exists and is not a directory, %s' % path)
    if io_util.IsCloudPath(path):
        logger.info('GetLogdirSubdirectories: Starting to list directories via glob-ing.')
        traversal_method = ListRecursivelyViaGlobbing
    else:
        logger.info('GetLogdirSubdirectories: Starting to list directories via walking.')
        traversal_method = ListRecursivelyViaWalking
    return (subdir for subdir, files in traversal_method(path) if any((IsTensorFlowEventsFile(f) for f in files)))