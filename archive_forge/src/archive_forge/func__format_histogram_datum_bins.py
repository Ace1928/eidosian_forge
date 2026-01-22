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
def _format_histogram_datum_bins(self, datum):
    """Formats a histogram datum's bins for client consumption.

        Args:
            datum: a DataProvider's TensorDatum.

        Returns:
            A list of `HistogramBin`s (see http_api.md).
        """
    numpy_list = datum.numpy.tolist()
    bins = [{'min': x[0], 'max': x[1], 'count': x[2]} for x in numpy_list]
    return bins