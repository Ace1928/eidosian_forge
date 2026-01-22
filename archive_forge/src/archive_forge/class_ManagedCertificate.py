from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedCertificate(_messages.Message):
    """A certificate managed by App Engine.

  Enums:
    StatusValueValuesEnum: Status of certificate management. Refers to the
      most recent certificate acquisition or renewal attempt.@OutputOnly

  Fields:
    lastRenewalTime: Time at which the certificate was last renewed. The
      renewal process is fully managed. Certificate renewal will automatically
      occur before the certificate expires. Renewal errors can be tracked via
      ManagementStatus.@OutputOnly
    status: Status of certificate management. Refers to the most recent
      certificate acquisition or renewal attempt.@OutputOnly
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Status of certificate management. Refers to the most recent
    certificate acquisition or renewal attempt.@OutputOnly

    Values:
      MANAGEMENT_STATUS_UNSPECIFIED: <no description>
      OK: Certificate was successfully obtained and inserted into the serving
        system.
      PENDING: Certificate is under active attempts to acquire or renew.
      FAILED_RETRYING_NOT_VISIBLE: Most recent renewal failed due to an
        invalid DNS setup and will be retried. Renewal attempts will continue
        to fail until the certificate domain's DNS configuration is fixed. The
        last successfully provisioned certificate may still be serving.
      FAILED_PERMANENT: All renewal attempts have been exhausted, likely due
        to an invalid DNS setup.
      FAILED_RETRYING_CAA_FORBIDDEN: Most recent renewal failed due to an
        explicit CAA record that does not include one of the in-use CAs
        (Google CA and Let's Encrypt). Renewals will continue to fail until
        the CAA is reconfigured. The last successfully provisioned certificate
        may still be serving.
      FAILED_RETRYING_CAA_CHECKING: Most recent renewal failed due to a CAA
        retrieval failure. This means that the domain's DNS provider does not
        properly handle CAA records, failing requests for CAA records when no
        CAA records are defined. Renewals will continue to fail until the DNS
        provider is changed or a CAA record is added for the given domain. The
        last successfully provisioned certificate may still be serving.
    """
        MANAGEMENT_STATUS_UNSPECIFIED = 0
        OK = 1
        PENDING = 2
        FAILED_RETRYING_NOT_VISIBLE = 3
        FAILED_PERMANENT = 4
        FAILED_RETRYING_CAA_FORBIDDEN = 5
        FAILED_RETRYING_CAA_CHECKING = 6
    lastRenewalTime = _messages.StringField(1)
    status = _messages.EnumField('StatusValueValuesEnum', 2)