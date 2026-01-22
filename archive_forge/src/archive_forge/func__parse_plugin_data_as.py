from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.util import tensor_util
def _parse_plugin_data_as(content, data_oneof_field):
    """Returns a data oneof's field from plugin_data.content.

    Raises HParamsError if the content doesn't have 'data_oneof_field' set or
    this file is incompatible with the version of the metadata stored.

    Args:
      content: The SummaryMetadata.plugin_data.content to use.
      data_oneof_field: string. The name of the data oneof field to return.
    """
    plugin_data = plugin_data_pb2.HParamsPluginData.FromString(content)
    if plugin_data.version != PLUGIN_DATA_VERSION:
        raise error.HParamsError('Only supports plugin_data version: %s; found: %s in: %s' % (PLUGIN_DATA_VERSION, plugin_data.version, plugin_data))
    if not plugin_data.HasField(data_oneof_field):
        raise error.HParamsError('Expected plugin_data.%s to be set. Got: %s' % (data_oneof_field, plugin_data))
    return getattr(plugin_data, data_oneof_field)