from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.custom_scalar import (
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.plugins.histogram import metadata as histograms_metadata
from tensorboard.plugins.hparams import metadata as hparams_metadata
from tensorboard.plugins.image import metadata as images_metadata
from tensorboard.plugins.mesh import metadata as mesh_metadata
from tensorboard.plugins.pr_curve import metadata as pr_curves_metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.plugins.text import metadata as text_metadata
from tensorboard.util import tensor_util
def _migrate_graph_event(old_event):
    result = event_pb2.Event()
    result.wall_time = old_event.wall_time
    result.step = old_event.step
    value = result.summary.value.add(tag=graphs_metadata.RUN_GRAPH_NAME)
    graph_bytes = old_event.graph_def
    value.tensor.CopyFrom(tensor_util.make_tensor_proto([graph_bytes]))
    value.metadata.plugin_data.plugin_name = graphs_metadata.PLUGIN_NAME
    value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    return (old_event, result)