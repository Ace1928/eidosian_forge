from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml.speech import flags
from googlecloudsdk.command_lib.ml.speech import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class RecognizeGA(base.Command):
    """Get transcripts of short (less than 60 seconds) audio from an audio file."""
    detailed_help = {'DESCRIPTION': 'Get a transcript of an audio file that is less than 60 seconds. You can use\nan audio file that is on your local drive or a Google Cloud Storage URL.\n\nIf the audio is longer than 60 seconds, you will get an error. Please use\n`{parent_command} recognize-long-running` instead.\n', 'EXAMPLES': "To get a transcript of an audio file 'my-recording.wav':\n\n    $ {command} 'my-recording.wav' --language-code=en-US\n\nTo get a transcript of an audio file in bucket 'gs://bucket/myaudio' with a\ncustom sampling rate and encoding that uses hints and filters profanity:\n\n    $ {command} 'gs://bucket/myaudio' \\\n        --language-code=es-ES --sample-rate=2200 --hints=Bueno \\\n        --encoding=OGG_OPUS --filter-profanity\n", 'API REFERENCE': 'This command uses the speech/v1 API. The full documentation for this API\ncan be found at: https://cloud.google.com/speech-to-text/docs/quickstart-protocol\n'}
    API_VERSION = 'v1'
    flags_mapper = flags.RecognizeArgsToRequestMapper()

    @classmethod
    def Args(cls, parser):
        parser.display_info.AddFormat('json')
        cls.flags_mapper.AddRecognizeArgsToParser(parser, cls.API_VERSION)

    def MakeRequest(self, args, messages):
        return messages.RecognizeRequest(audio=util.GetRecognitionAudioFromPath(args.audio, self.API_VERSION), config=self.flags_mapper.MakeRecognitionConfig(args, messages))

    def Run(self, args):
        """Run 'ml speech recognize'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      Nothing.
    """
        client = apis.GetClientInstance(util.SPEECH_API, self.API_VERSION)
        self._request = self.MakeRequest(args, client.MESSAGES_MODULE)
        return client.speech.Recognize(self._request)

    def Epilog(self, unused_resources_were_displayed):
        util.MaybePrintSttUiLink(self._request)