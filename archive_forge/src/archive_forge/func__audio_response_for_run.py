import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.audio import metadata
def _audio_response_for_run(self, ctx, experiment, run, tag, sample):
    """Builds a JSON-serializable object with information about audio.

        Args:
          run: The name of the run.
          tag: The name of the tag the audio entries all belong to.
          sample: The zero-indexed sample of the audio sample for which to
          retrieve information. For instance, setting `sample` to `2` will
            fetch information about only the third audio clip of each batch,
            and steps with fewer than three audio clips will be omitted from
            the results.

        Returns:
          A list of dictionaries containing the wall time, step, label,
          content type, and query string for each audio entry.
        """
    all_audio = self._data_provider.read_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, downsample=self._downsample_to, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]))
    audio = all_audio.get(run, {}).get(tag, None)
    if audio is None:
        raise errors.NotFoundError('No audio data for run=%r, tag=%r' % (run, tag))
    content_type = self._get_mime_type(ctx, experiment, run, tag)
    response = []
    for datum in audio:
        if len(datum.values) < sample:
            continue
        query = urllib.parse.urlencode({'blob_key': datum.values[sample].blob_key, 'content_type': content_type})
        response.append({'wall_time': datum.wall_time, 'label': '', 'step': datum.step, 'contentType': content_type, 'query': query})
    return response