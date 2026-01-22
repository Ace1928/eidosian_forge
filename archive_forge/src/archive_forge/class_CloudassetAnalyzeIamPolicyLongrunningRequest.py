from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetAnalyzeIamPolicyLongrunningRequest(_messages.Message):
    """A CloudassetAnalyzeIamPolicyLongrunningRequest object.

  Fields:
    analyzeIamPolicyLongrunningRequest: A AnalyzeIamPolicyLongrunningRequest
      resource to be passed as the request body.
    scope: Required. The relative name of the root asset. Only resources and
      IAM policies within the scope will be analyzed. This can only be an
      organization number (such as "organizations/123"), a folder number (such
      as "folders/123"), a project ID (such as "projects/my-project-id"), or a
      project number (such as "projects/12345"). To know how to get
      organization ID, visit [here ](https://cloud.google.com/resource-
      manager/docs/creating-managing-
      organization#retrieving_your_organization_id). To know how to get folder
      or project ID, visit [here ](https://cloud.google.com/resource-
      manager/docs/creating-managing-
      folders#viewing_or_listing_folders_and_projects).
  """
    analyzeIamPolicyLongrunningRequest = _messages.MessageField('AnalyzeIamPolicyLongrunningRequest', 1)
    scope = _messages.StringField(2, required=True)