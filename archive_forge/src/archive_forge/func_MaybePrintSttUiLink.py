from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from six.moves import urllib
def MaybePrintSttUiLink(request):
    """Print Url to the Speech-to-text UI console for given recognize request."""
    if console_io.IsRunFromShellScript() or properties.VALUES.core.disable_prompts.GetBool():
        return
    audio_uri = request.audio.uri
    if not audio_uri:
        return
    payload = {'audio': urllib.parse.quote_plus(audio_uri[5:] if audio_uri.startswith('gs://') else audio_uri), 'encoding': request.config.encoding, 'model': request.config.model, 'locale': request.config.languageCode, 'sampling': request.config.sampleRateHertz, 'channels': request.config.audioChannelCount, 'enhanced': request.config.useEnhanced}
    params = ';'.join(('{}={}'.format(key, value) for key, value in sorted(payload.items()) if value and 'unspecified' not in str(value).lower()))
    url_tuple = ('https', 'console.cloud.google.com', '/speech/transcriptions/create', params, '', '')
    target_url = urllib.parse.urlunparse(url_tuple)
    log.status.Print('Try this using the Speech-to-Text UI at {}'.format(target_url))