import os.path
import time
import warnings
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
from tensorflow.python.util.tf_export import tf_export
def _write_plugin_assets(self, graph):
    plugin_assets = plugin_asset.get_all_plugin_assets(graph)
    logdir = self.event_writer.get_logdir()
    for asset_container in plugin_assets:
        plugin_name = asset_container.plugin_name
        plugin_dir = os.path.join(logdir, _PLUGINS_DIR, plugin_name)
        gfile.MakeDirs(plugin_dir)
        assets = asset_container.assets()
        for asset_name, content in assets.items():
            asset_path = os.path.join(plugin_dir, asset_name)
            with gfile.Open(asset_path, 'w') as f:
                f.write(content)