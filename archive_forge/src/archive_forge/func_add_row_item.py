import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import keras_util
from tensorboard.plugins.graph import metadata
from tensorboard.util import tb_logging
def add_row_item(run, tag=None):
    run_item = result.setdefault(run, {'run': run, 'tags': {}, 'run_graph': False})
    tag_item = None
    if tag:
        tag_item = run_item.get('tags').setdefault(tag, {'tag': tag, 'conceptual_graph': False, 'op_graph': False, 'profile': False})
    return (run_item, tag_item)