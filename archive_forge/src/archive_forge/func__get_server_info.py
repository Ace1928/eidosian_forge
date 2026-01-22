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
def _get_server_info(api_endpoint=None):
    plugins = ['scalars']
    if api_endpoint:
        return server_info_lib.create_server_info(DEFAULT_ORIGIN, api_endpoint, plugins)
    return server_info_lib.fetch_server_info(DEFAULT_ORIGIN, plugins)