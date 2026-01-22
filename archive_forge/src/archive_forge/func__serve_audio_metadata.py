import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.audio import metadata
@wrappers.Request.application
def _serve_audio_metadata(self, request):
    """Given a tag and list of runs, serve a list of metadata for audio.

        Note that the actual audio data are not sent; instead, we respond
        with URLs to the audio. The frontend should treat these URLs as
        opaque and should not try to parse information about them or
        generate them itself, as the format may change.

        Args:
          request: A werkzeug.wrappers.Request object.

        Returns:
          A werkzeug.Response application.
        """
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    tag = request.args.get('tag')
    run = request.args.get('run')
    sample = int(request.args.get('sample', 0))
    response = self._audio_response_for_run(ctx, experiment, run, tag, sample)
    return http_util.Respond(request, response, 'application/json')