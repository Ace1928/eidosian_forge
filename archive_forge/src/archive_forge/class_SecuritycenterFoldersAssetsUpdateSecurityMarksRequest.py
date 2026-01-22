from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersAssetsUpdateSecurityMarksRequest(_messages.Message):
    """A SecuritycenterFoldersAssetsUpdateSecurityMarksRequest object.

  Fields:
    googleCloudSecuritycenterV2SecurityMarks: A
      GoogleCloudSecuritycenterV2SecurityMarks resource to be passed as the
      request body.
    name: The relative resource name of the SecurityMarks. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me The following list shows some examples: +
      `organizations/{organization_id}/assets/{asset_id}/securityMarks` + `org
      anizations/{organization_id}/sources/{source_id}/findings/{finding_id}/s
      ecurityMarks` + `organizations/{organization_id}/sources/{source_id}/loc
      ations/{location}/findings/{finding_id}/securityMarks`
    updateMask: The FieldMask to use when updating the security marks
      resource. The field mask must not contain duplicate fields. If empty or
      set to "marks", all marks will be replaced. Individual marks can be
      updated using "marks.".
  """
    googleCloudSecuritycenterV2SecurityMarks = _messages.MessageField('GoogleCloudSecuritycenterV2SecurityMarks', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)