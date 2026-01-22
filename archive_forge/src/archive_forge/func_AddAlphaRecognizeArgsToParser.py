from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAlphaRecognizeArgsToParser(self, parser, api_version):
    """Add alpha arguments."""
    meta_args = parser.add_group(required=False, help='Description of audio data to be recognized. Note that the Google Cloud Speech-to-text-api does not use this information, and only passes it through back into response.')
    meta_args.add_argument('--naics-code', action=MakeDeprecatedRecgonitionFlagAction('naics-code'), type=int, help='The industry vertical to which this speech recognition request most closely applies.')
    self._original_media_type_mapper = GetOriginalMediaTypeMapper(api_version)
    self._original_media_type_mapper.choice_arg.AddToParser(meta_args)
    self._interaction_type_mapper = GetInteractionTypeMapper(api_version)
    self._interaction_type_mapper.choice_arg.AddToParser(meta_args)
    self._microphone_distance_type_mapper = GetMicrophoneDistanceTypeMapper(api_version)
    self._microphone_distance_type_mapper.choice_arg.AddToParser(meta_args)
    self._device_type_mapper = GetRecordingDeviceTypeMapper(api_version)
    self._device_type_mapper.choice_arg.AddToParser(meta_args)
    meta_args.add_argument('--recording-device-name', action=MakeDeprecatedRecgonitionFlagAction('recording-device-name'), help='The device used to make the recording.  Examples: `Nexus 5X`, `Polycom SoundStation IP 6000`')
    meta_args.add_argument('--original-mime-type', action=MakeDeprecatedRecgonitionFlagAction('original-mime-type'), help='Mime type of the original audio file. Examples: `audio/m4a`,  `audio/mp3`.')
    meta_args.add_argument('--audio-topic', action=MakeDeprecatedRecgonitionFlagAction('audio-topic'), help='Description of the content, e.g. "Recordings of federal supreme court hearings from 2012".')