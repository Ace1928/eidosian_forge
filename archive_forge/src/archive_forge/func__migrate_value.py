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
def _migrate_value(value, initial_metadata):
    """Convert an old value to a stream of new values. May mutate."""
    metadata = initial_metadata.get(value.tag)
    initial = False
    if metadata is None:
        initial = True
        metadata = summary_pb2.SummaryMetadata()
        metadata.CopyFrom(value.metadata)
        initial_metadata[value.tag] = metadata
    if metadata.data_class != summary_pb2.DATA_CLASS_UNKNOWN:
        return (value,)
    plugin_name = metadata.plugin_data.plugin_name
    if plugin_name == histograms_metadata.PLUGIN_NAME:
        return _migrate_histogram_value(value)
    if plugin_name == images_metadata.PLUGIN_NAME:
        return _migrate_image_value(value)
    if plugin_name == audio_metadata.PLUGIN_NAME:
        return _migrate_audio_value(value)
    if plugin_name == scalars_metadata.PLUGIN_NAME:
        return _migrate_scalar_value(value)
    if plugin_name == text_metadata.PLUGIN_NAME:
        return _migrate_text_value(value)
    if plugin_name == hparams_metadata.PLUGIN_NAME:
        return _migrate_hparams_value(value)
    if plugin_name == pr_curves_metadata.PLUGIN_NAME:
        return _migrate_pr_curve_value(value)
    if plugin_name == mesh_metadata.PLUGIN_NAME:
        return _migrate_mesh_value(value)
    if plugin_name == custom_scalars_metadata.PLUGIN_NAME:
        return _migrate_custom_scalars_value(value)
    if plugin_name in [graphs_metadata.PLUGIN_NAME_RUN_METADATA, graphs_metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH, graphs_metadata.PLUGIN_NAME_KERAS_MODEL]:
        return _migrate_graph_sub_plugin_value(value)
    return (value,)