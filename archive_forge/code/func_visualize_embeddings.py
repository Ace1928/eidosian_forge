import os
from google.protobuf import text_format as _text_format
from tensorboard.compat import tf
from tensorboard.plugins.projector import metadata as _metadata
from tensorboard.plugins.projector.projector_config_pb2 import (  # noqa: F401
from tensorboard.plugins.projector.projector_config_pb2 import (  # noqa: F401
from tensorboard.plugins.projector.projector_config_pb2 import (  # noqa: F401
def visualize_embeddings(logdir, config):
    """Stores a config file used by the embedding projector.

    Args:
      logdir: Directory into which to store the config file, as a `str`.
        For compatibility, can also be a `tf.compat.v1.summary.FileWriter`
        object open at the desired logdir.
      config: `tf.contrib.tensorboard.plugins.projector.ProjectorConfig`
        proto that holds the configuration for the projector such as paths to
        checkpoint files and metadata files for the embeddings. If
        `config.model_checkpoint_path` is none, it defaults to the
        `logdir` used by the summary_writer.

    Raises:
      ValueError: If the summary writer does not have a `logdir`.
    """
    logdir = getattr(logdir, 'get_logdir', lambda: logdir)()
    if logdir is None:
        raise ValueError('Expected logdir to be a path, but got None')
    config_pbtxt = _text_format.MessageToString(config)
    path = os.path.join(logdir, _metadata.PROJECTOR_FILENAME)
    with tf.io.gfile.GFile(path, 'w') as f:
        f.write(config_pbtxt)