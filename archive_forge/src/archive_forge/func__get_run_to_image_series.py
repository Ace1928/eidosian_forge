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
def _get_run_to_image_series(self, ctx, experiment, tag, sample, runs):
    """Builds a run-to-image-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            sample: zero-indexed integer for the requested sample.
            runs: optional list of run names as strings.

        Returns:
            A `RunToSeries` dict (see http_api.md).
        """
    mapping = self._data_provider.read_blob_sequences(ctx, experiment_id=experiment, plugin_name=image_metadata.PLUGIN_NAME, downsample=self._plugin_downsampling['images'], run_tag_filter=provider.RunTagFilter(runs, tags=[tag]))
    run_to_series = {}
    for result_run, tag_data in mapping.items():
        if tag not in tag_data:
            continue
        blob_sequence_datum_list = tag_data[tag]
        series = _format_image_blob_sequence_datum(blob_sequence_datum_list, sample)
        if series:
            run_to_series[result_run] = series
    return run_to_series