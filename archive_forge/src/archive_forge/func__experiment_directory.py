import base64
import contextlib
import errno
import grpc
import json
import os
import string
import time
import numpy as np
from tensorboard.uploader.proto import blob_pb2
from tensorboard.uploader.proto import experiment_pb2
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader import util
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _experiment_directory(base_dir, experiment_id):
    bad_chars = frozenset(experiment_id) - _FILENAME_SAFE_CHARS
    if bad_chars:
        raise RuntimeError('Unexpected characters ({bad_chars!r}) in experiment ID {eid!r}'.format(bad_chars=sorted(bad_chars), eid=experiment_id))
    return os.path.join(base_dir, 'experiment_%s' % experiment_id)