from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttestationModeValueValuesEnum(_messages.Enum):
    """Optional. Configures the behavior for attesting results.

    Values:
      ATTESTATION_MODE_UNSPECIFIED: Unspecified. Results are not attested.
      GENERATE_DEPLOY: Generate and return deploy attestations in DSEE form.
    """
    ATTESTATION_MODE_UNSPECIFIED = 0
    GENERATE_DEPLOY = 1