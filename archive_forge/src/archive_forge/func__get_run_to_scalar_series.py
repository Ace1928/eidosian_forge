import collections
import imghdr
import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _get_run_to_scalar_series(self, ctx, experiment, tag, runs):
    """Builds a run-to-scalar-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            runs: optional list of run names as strings.

        Returns:
            A map from string run names to `ScalarStepDatum` (see http_api.md).
        """
    mapping = self._data_provider.read_scalars(ctx, experiment_id=experiment, plugin_name=scalar_metadata.PLUGIN_NAME, downsample=self._plugin_downsampling['scalars'], run_tag_filter=provider.RunTagFilter(runs=runs, tags=[tag]))
    run_to_series = {}
    for result_run, tag_data in mapping.items():
        if tag not in tag_data:
            continue
        values = [{'wallTime': datum.wall_time, 'step': datum.step, 'value': datum.value} for datum in tag_data[tag]]
        run_to_series[result_run] = values
    return run_to_series