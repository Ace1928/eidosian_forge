from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class VoiceSelectionParams(proto.Message):
    """Description of which voice to use for a synthesis request.

    Attributes:
        language_code (str):
            Required. The language (and potentially also the region) of
            the voice expressed as a
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag, e.g. "en-US". This should not include a script
            tag (e.g. use "cmn-cn" rather than "cmn-Hant-cn"), because
            the script will be inferred from the input provided in the
            SynthesisInput. The TTS service will use this parameter to
            help choose an appropriate voice. Note that the TTS service
            may choose a voice with a slightly different language code
            than the one selected; it may substitute a different region
            (e.g. using en-US rather than en-CA if there isn't a
            Canadian voice available), or even a different language,
            e.g. using "nb" (Norwegian Bokmal) instead of "no"
            (Norwegian)".
        name (str):
            The name of the voice. If not set, the service will choose a
            voice based on the other parameters such as language_code
            and gender.
        ssml_gender (google.cloud.texttospeech_v1.types.SsmlVoiceGender):
            The preferred gender of the voice. If not set, the service
            will choose a voice based on the other parameters such as
            language_code and name. Note that this is only a preference,
            not requirement; if a voice of the appropriate gender is not
            available, the synthesizer should substitute a voice with a
            different gender rather than failing the request.
        custom_voice (google.cloud.texttospeech_v1.types.CustomVoiceParams):
            The configuration for a custom voice. If
            [CustomVoiceParams.model] is set, the service will choose
            the custom voice matching the specified configuration.
    """
    language_code: str = proto.Field(proto.STRING, number=1)
    name: str = proto.Field(proto.STRING, number=2)
    ssml_gender: 'SsmlVoiceGender' = proto.Field(proto.ENUM, number=3, enum='SsmlVoiceGender')
    custom_voice: 'CustomVoiceParams' = proto.Field(proto.MESSAGE, number=4, message='CustomVoiceParams')