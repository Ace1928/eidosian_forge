from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Access(_messages.Message):
    """Represents an access event.

  Fields:
    callerIp: Caller's IP address, such as "1.1.1.1".
    callerIpGeo: The caller IP's geolocation, which identifies where the call
      came from.
    methodName: The method that the service account called, e.g.
      "SetIamPolicy".
    principalEmail: Associated email, such as "foo@google.com". The email
      address of the authenticated user or a service account acting on behalf
      of a third party principal making the request. For third party identity
      callers, the `principal_subject` field is populated instead of this
      field. For privacy reasons, the principal email address is sometimes
      redacted. For more information, see [Caller identities in audit
      logs](https://cloud.google.com/logging/docs/audit#user-id).
    principalSubject: A string that represents the principal_subject that is
      associated with the identity. Unlike `principal_email`,
      `principal_subject` supports principals that aren't associated with
      email addresses, such as third party principals. For most identities,
      the format is `principal://iam.googleapis.com/{identity pool
      name}/subject/{subject}`. Some GKE identities, such as GKE_WORKLOAD,
      FREEFORM, and GKE_HUB_WORKLOAD, still use the legacy format
      `serviceAccount:{identity pool name}[{subject}]`.
    serviceAccountDelegationInfo: The identity delegation history of an
      authenticated service account that made the request. The
      `serviceAccountDelegationInfo[]` object contains information about the
      real authorities that try to access Google Cloud resources by delegating
      on a service account. When multiple authorities are present, they are
      guaranteed to be sorted based on the original ordering of the identity
      delegation events.
    serviceAccountKeyName: The name of the service account key that was used
      to create or exchange credentials when authenticating the service
      account that made the request. This is a scheme-less URI full resource
      name. For example: "//iam.googleapis.com/projects/{PROJECT_ID}/serviceAc
      counts/{ACCOUNT}/keys/{key}".
    serviceName: This is the API service that the service account made a call
      to, e.g. "iam.googleapis.com"
    userAgent: The caller's user agent string associated with the finding.
    userAgentFamily: Type of user agent associated with the finding. For
      example, an operating system shell or an embedded or standalone
      application.
    userName: A string that represents a username. The username provided
      depends on the type of the finding and is likely not an IAM principal.
      For example, this can be a system username if the finding is related to
      a virtual machine, or it can be an application login username.
  """
    callerIp = _messages.StringField(1)
    callerIpGeo = _messages.MessageField('GoogleCloudSecuritycenterV2Geolocation', 2)
    methodName = _messages.StringField(3)
    principalEmail = _messages.StringField(4)
    principalSubject = _messages.StringField(5)
    serviceAccountDelegationInfo = _messages.MessageField('GoogleCloudSecuritycenterV2ServiceAccountDelegationInfo', 6, repeated=True)
    serviceAccountKeyName = _messages.StringField(7)
    serviceName = _messages.StringField(8)
    userAgent = _messages.StringField(9)
    userAgentFamily = _messages.StringField(10)
    userName = _messages.StringField(11)