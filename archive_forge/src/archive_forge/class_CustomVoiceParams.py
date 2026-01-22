from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class CustomVoiceParams(proto.Message):
    """Description of the custom voice to be synthesized.

    Attributes:
        model (str):
            Required. The name of the AutoML model that
            synthesizes the custom voice.
        reported_usage (google.cloud.texttospeech_v1.types.CustomVoiceParams.ReportedUsage):
            Optional. Deprecated. The usage of the
            synthesized audio to be reported.
    """

    class ReportedUsage(proto.Enum):
        """Deprecated. The usage of the synthesized audio. Usage does
        not affect billing.

        Values:
            REPORTED_USAGE_UNSPECIFIED (0):
                Request with reported usage unspecified will
                be rejected.
            REALTIME (1):
                For scenarios where the synthesized audio is
                not downloadable and can only be used once. For
                example, real-time request in IVR system.
            OFFLINE (2):
                For scenarios where the synthesized audio is
                downloadable and can be reused. For example, the
                synthesized audio is downloaded, stored in
                customer service system and played repeatedly.
        """
        REPORTED_USAGE_UNSPECIFIED = 0
        REALTIME = 1
        OFFLINE = 2
    model: str = proto.Field(proto.STRING, number=1)
    reported_usage: ReportedUsage = proto.Field(proto.ENUM, number=3, enum=ReportedUsage)