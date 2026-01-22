from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
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