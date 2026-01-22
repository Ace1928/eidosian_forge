import collections
import os
import re
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def ListRecursivelyViaWalking(top):
    """Walks a directory tree, yielding (dir_path, file_paths) tuples.

    For each of `top` and its subdirectories, yields a tuple containing the path
    to the directory and the path to each of the contained files.  Note that
    unlike os.Walk()/tf.io.gfile.walk()/ListRecursivelyViaGlobbing, this does not
    list subdirectories. The file paths are all absolute. If the directory does
    not exist, this yields nothing.

    Walking may be incredibly slow on certain file systems.

    Args:
      top: A path to a directory.

    Yields:
      A (dir_path, file_paths) tuple for each directory/subdirectory.
    """
    for dir_path, _, filenames in tf.io.gfile.walk(top, topdown=True):
        yield (dir_path, (os.path.join(dir_path, filename) for filename in filenames))