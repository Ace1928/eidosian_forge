from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseChannelValueValuesEnum(_messages.Enum):
    """Release channel influences the timing and frequency of new updates to
    the Apigee runtimes instances of the organization. It can be either
    STABLE, REGULAR, or RAPID. It can be selected during creation of the
    Organization and it can also be updated later on. Each channel has its own
    combination of release frequency and stability expectations. The RAPID
    channel will get updates early and more often. The REGULAR channel will
    get updates after being validated in the RAPID channel for some time. The
    STABLE channel will get updates after being validated in the REGULAR
    channel for some time.

    Values:
      RELEASE_CHANNEL_UNSPECIFIED: Release channel not specified.
      STABLE: Stable release channel.
      REGULAR: Regular release channel.
      RAPID: Rapid release channel.
    """
    RELEASE_CHANNEL_UNSPECIFIED = 0
    STABLE = 1
    REGULAR = 2
    RAPID = 3