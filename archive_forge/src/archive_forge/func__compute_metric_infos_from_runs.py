import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _compute_metric_infos_from_runs(self, ctx, experiment_id, hparams_run_to_tag_to_content):
    session_runs = set((run for run, tags in hparams_run_to_tag_to_content.items() if metadata.SESSION_START_INFO_TAG in tags))
    return (api_pb2.MetricInfo(name=api_pb2.MetricName(group=group, tag=tag)) for tag, group in self._compute_metric_names(ctx, experiment_id, session_runs))