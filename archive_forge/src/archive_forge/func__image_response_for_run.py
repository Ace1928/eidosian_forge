import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
def _image_response_for_run(self, ctx, experiment, run, tag, sample):
    """Builds a JSON-serializable object with information about images.

        Args:
          run: The name of the run.
          tag: The name of the tag the images all belong to.
          sample: The zero-indexed sample of the image for which to retrieve
            information. For instance, setting `sample` to `2` will fetch
            information about only the third image of each batch. Steps with
            fewer than three images will be omitted from the results.

        Returns:
          A list of dictionaries containing the wall time, step, and URL
          for each image.

        Raises:
          KeyError, NotFoundError: If no image data exists for the given
            parameters.
        """
    all_images = self._data_provider.read_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, downsample=self._downsample_to, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]))
    images = all_images.get(run, {}).get(tag, None)
    if images is None:
        raise errors.NotFoundError('No image data for run=%r, tag=%r' % (run, tag))
    return [{'wall_time': datum.wall_time, 'step': datum.step, 'query': self._data_provider_query(datum.values[sample + 2])} for datum in images if len(datum.values) - 2 > sample]