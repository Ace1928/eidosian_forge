import numpy as np
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util
def _collect_tensor_events(self, request, step=None):
    """Collects list of tensor events based on request."""
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    tag = request.args.get('tag')
    tensor_events = []
    for instance_tag in self._instance_tags(ctx, experiment, run, tag):
        tensors = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[instance_tag]), downsample=self._downsample_to)[run][instance_tag]
        meta = self._instance_tag_metadata(ctx, experiment, run, instance_tag)
        tensor_events += [(meta, tensor) for tensor in tensors]
    if step is not None:
        tensor_events = [event for event in tensor_events if event[1].step == step]
    else:
        tensor_events = sorted(tensor_events, key=lambda tensor_data: tensor_data[1].step)
    return tensor_events