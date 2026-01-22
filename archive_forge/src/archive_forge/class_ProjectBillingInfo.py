from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ProjectBillingInfo(_messages.Message):
    """Encapsulation of billing information for a Google Cloud Console project.
  A project has at most one associated billing account at a time (but a
  billing account can be assigned to multiple projects).

  Fields:
    billingAccountName: The resource name of the billing account associated
      with the project, if any. For example,
      `billingAccounts/012345-567890-ABCDEF`.
    billingEnabled: Output only. True if the project is associated with an
      open billing account, to which usage on the project is charged. False if
      the project is associated with a closed billing account, or no billing
      account at all, and therefore cannot use paid services.
    name: Output only. The resource name for the `ProjectBillingInfo`; has the
      form `projects/{project_id}/billingInfo`. For example, the resource name
      for the billing information for project `tokyo-rain-123` would be
      `projects/tokyo-rain-123/billingInfo`.
    projectId: Output only. The ID of the project that this
      `ProjectBillingInfo` represents, such as `tokyo-rain-123`. This is a
      convenience field so that you don't need to parse the `name` field to
      obtain a project ID.
  """
    billingAccountName = _messages.StringField(1)
    billingEnabled = _messages.BooleanField(2)
    name = _messages.StringField(3)
    projectId = _messages.StringField(4)