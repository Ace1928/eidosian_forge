import collections
import os
import re
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def ListRecursivelyViaGlobbing(top):
    """Recursively lists all files within the directory.

    This method does not list subdirectories (in addition to regular files), and
    the file paths are all absolute. If the directory does not exist, this yields
    nothing.

    This method does so by glob-ing deeper and deeper directories, ie
    foo/*, foo/*/*, foo/*/*/* and so on until all files are listed. All file
    paths are absolute, and this method lists subdirectories too.

    For certain file systems, globbing via this method may prove significantly
    faster than recursively walking a directory. Specifically, TF file systems
    that implement TensorFlow's FileSystem.GetMatchingPaths method could save
    costly disk reads by using this method. However, for other file systems, this
    method might prove slower because the file system performs a walk per call to
    glob (in which case it might as well just perform 1 walk).

    Args:
      top: A path to a directory.

    Yields:
      A (dir_path, file_paths) tuple for each directory/subdirectory.
    """
    current_glob_string = os.path.join(_EscapeGlobCharacters(top), '*')
    level = 0
    while True:
        logger.info('GlobAndListFiles: Starting to glob level %d', level)
        glob = tf.io.gfile.glob(current_glob_string)
        logger.info('GlobAndListFiles: %d files glob-ed at level %d', len(glob), level)
        if not glob:
            return
        pairs = collections.defaultdict(list)
        for file_path in glob:
            pairs[os.path.dirname(file_path)].append(file_path)
        for dir_name, file_paths in pairs.items():
            yield (dir_name, tuple(file_paths))
        if len(pairs) == 1:
            current_glob_string = os.path.join(list(pairs.keys())[0], '*')
        current_glob_string = os.path.join(current_glob_string, '*')
        level += 1