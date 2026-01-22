import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _compute_hparam_infos(self, hparams_run_to_tag_to_content):
    """Computes a list of api_pb2.HParamInfo from the current run, tag
        info.

        Finds all the SessionStartInfo messages and collects the hparams values
        appearing in each one. For each hparam attempts to deduce a type that fits
        all its values. Finally, sets the 'domain' of the resulting HParamInfo
        to be discrete if the type is string or boolean.

        Returns:
          A list of api_pb2.HParamInfo messages.
        """
    hparams = collections.defaultdict(list)
    for tag_to_content in hparams_run_to_tag_to_content.values():
        if metadata.SESSION_START_INFO_TAG not in tag_to_content:
            continue
        start_info = metadata.parse_session_start_info_plugin_data(tag_to_content[metadata.SESSION_START_INFO_TAG])
        for name, value in start_info.hparams.items():
            hparams[name].append(value)
    result = []
    for name, values in hparams.items():
        hparam_info = self._compute_hparam_info_from_values(name, values)
        if hparam_info is not None:
            result.append(hparam_info)
    return result