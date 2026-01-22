from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml.speech import flags
from googlecloudsdk.command_lib.ml.speech import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class RecognizeLongRunningGA(base.Command):
    """Get transcripts of longer audio from an audio file."""
    detailed_help = {'DESCRIPTION': 'Get a transcript of audio up to 80 minutes in length. If the audio is\nunder 60 seconds, you may also use `{parent_command} recognize` to\nanalyze it.\n', 'EXAMPLES': "To block the command from completing until analysis is finished, run:\n\n  $ {command} AUDIO_FILE --language-code=LANGUAGE_CODE --sample-rate=SAMPLE_RATE\n\nYou can also receive an operation as the result of the command by running:\n\n  $ {command} AUDIO_FILE --language-code=LANGUAGE_CODE --sample-rate=SAMPLE_RATE --async\n\nThis will return information about an operation. To get information about the\noperation, run:\n\n  $ {parent_command} operations describe OPERATION_ID\n\nTo poll the operation until it's complete, run:\n\n  $ {parent_command} operations wait OPERATION_ID\n", 'API REFERENCE': 'This command uses the speech/v1 API. The full documentation for this API\ncan be found at: https://cloud.google.com/speech-to-text/docs/quickstart-protocol\n'}
    API_VERSION = 'v1'
    flags_mapper = flags.RecognizeArgsToRequestMapper()

    @classmethod
    def Args(cls, parser):
        parser.display_info.AddFormat('json')
        cls.flags_mapper.AddRecognizeArgsToParser(parser, cls.API_VERSION)
        base.ASYNC_FLAG.AddToParser(parser)
        parser.add_argument('--output-uri', type=util.ValidateOutputUri, help='Location to which the results should be written. Must be a Google Cloud Storage URI.')

    def MakeRequest(self, args, messages):
        request = messages.LongRunningRecognizeRequest(audio=util.GetRecognitionAudioFromPath(args.audio, self.API_VERSION), config=self.flags_mapper.MakeRecognitionConfig(args, messages))
        if args.output_uri is not None:
            request.outputConfig = messages.TranscriptOutputConfig(gcsUri=args.output_uri)
        return request

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
        operation = client.speech.Longrunningrecognize(self._request)
        if args.async_:
            return operation
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(client.operations, lambda x: x), operation.name, 'Waiting for [{}] to complete. This may take several minutes.'.format(operation.name), wait_ceiling_ms=OPERATION_TIMEOUT_MS)

    def Epilog(self, unused_resources_were_displayed):
        util.MaybePrintSttUiLink(self._request)