import os.path
from tensorboard.compat import tf
def _IsDirectory(parent, item):
    """Helper that returns if parent/item is a directory."""
    return tf.io.gfile.isdir(os.path.join(parent, item))