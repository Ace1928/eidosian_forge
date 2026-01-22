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
def _format_image_blob_sequence_datum(sorted_datum_list, sample):
    """Formats image metadata from a list of BlobSequenceDatum's for clients.

    This expects that frontend clients need to access images based on the
    run+tag+sample.

    Args:
        sorted_datum_list: a list of DataProvider's `BlobSequenceDatum`, sorted by
            step. This can be produced via DataProvider's `read_blob_sequences`.
        sample: zero-indexed integer for the requested sample.

    Returns:
        A list of `ImageStepDatum` (see http_api.md).
    """
    index = sample + 2
    step_data = []
    for datum in sorted_datum_list:
        if len(datum.values) <= index:
            continue
        step_data.append({'step': datum.step, 'wallTime': datum.wall_time, 'imageId': datum.values[index].blob_key})
    return step_data