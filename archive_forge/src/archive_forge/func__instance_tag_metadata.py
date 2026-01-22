import numpy as np
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util
def _instance_tag_metadata(self, ctx, experiment, run, instance_tag):
    """Gets the `MeshPluginData` proto for an instance tag."""
    results = self._data_provider.list_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[instance_tag]))
    content = results[run][instance_tag].plugin_content
    return metadata.parse_plugin_metadata(content)