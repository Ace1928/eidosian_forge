import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _test_run_tag(self, run_tag_filter, run, tag):
    runs = run_tag_filter.runs
    if runs is not None and run not in runs:
        return False
    tags = run_tag_filter.tags
    if tags is not None and tag not in tags:
        return False
    return True