import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _compute_experiment_from_runs(self, ctx, experiment_id, hparams_run_to_tag_to_content):
    """Computes a minimal Experiment protocol buffer by scanning the runs.

        Returns None if there are no hparam infos logged.
        """
    hparam_infos = self._compute_hparam_infos(hparams_run_to_tag_to_content)
    if hparam_infos:
        metric_infos = self._compute_metric_infos(ctx, experiment_id, hparams_run_to_tag_to_content)
    else:
        metric_infos = []
    if not hparam_infos and (not metric_infos):
        return None
    return api_pb2.Experiment(hparam_infos=hparam_infos, metric_infos=metric_infos)