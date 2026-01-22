import sys
import time
import grpc
from tensorboard.data.experimental import base_experiment
from tensorboard.data.experimental import utils as experimental_utils
from tensorboard.uploader import auth
from tensorboard.uploader import util
from tensorboard.uploader import server_info as server_info_lib
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader.proto import export_service_pb2_grpc
from tensorboard.uploader.proto import server_info_pb2
from tensorboard.util import grpc_util
def _handle_server_info(info):
    compat = info.compatibility
    if compat.verdict == server_info_pb2.VERDICT_WARN:
        sys.stderr.write('Warning [from server]: %s\n' % compat.details)
        sys.stderr.flush()
    elif compat.verdict == server_info_pb2.VERDICT_ERROR:
        raise ValueError('Error [from server]: %s' % compat.details)