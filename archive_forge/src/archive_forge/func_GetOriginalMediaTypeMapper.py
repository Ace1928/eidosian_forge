from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetOriginalMediaTypeMapper(version):
    messages = apis.GetMessagesModule(util.SPEECH_API, version)
    return arg_utils.ChoiceEnumMapper('--original-media-type', messages.RecognitionMetadata.OriginalMediaTypeValueValuesEnum, action=MakeDeprecatedRecgonitionFlagAction('original-media-type'), custom_mappings={'AUDIO': ('audio', 'The speech data is an audio recording.'), 'VIDEO': ('video', 'The speech data originally recorded on a video.')}, help_str='The media type of the original audio conversation.', include_filter=lambda x: not x.endswith('UNSPECIFIED'))