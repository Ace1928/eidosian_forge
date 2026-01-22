from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PscInstanceConfig(_messages.Message):
    """PscInstanceConfig contains PSC related configuration at an instance
  level.

  Fields:
    allowedConsumerProjects: Optional. List of consumer projects that are
      allowed to create PSC endpoints to service-attachments to this instance.
    pscDnsName: Output only. The DNS name of the instance for PSC
      connectivity. Name convention: ...alloydb-psc.goog
    serviceAttachmentLink: Output only. The service attachment created when
      Private Service Connect (PSC) is enabled for the instance. The name of
      the resource will be in the format of
      `projects//regions//serviceAttachments/`
  """
    allowedConsumerProjects = _messages.StringField(1, repeated=True)
    pscDnsName = _messages.StringField(2)
    serviceAttachmentLink = _messages.StringField(3)