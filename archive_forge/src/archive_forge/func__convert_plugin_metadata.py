import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _convert_plugin_metadata(self, data_provider_output):
    return {run: {tag: time_series.plugin_content for tag, time_series in tag_to_time_series.items()} for run, tag_to_time_series in data_provider_output.items()}