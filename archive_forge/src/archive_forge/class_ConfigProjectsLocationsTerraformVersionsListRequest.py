from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsTerraformVersionsListRequest(_messages.Message):
    """A ConfigProjectsLocationsTerraformVersionsListRequest object.

  Fields:
    filter: Optional. Lists the TerraformVersions that match the filter
      expression. A filter expression filters the resources listed in the
      response. The expression must be of the form '{field} {operator}
      {value}' where operators: '<', '>', '<=', '>=', '!=', '=', ':' are
      supported (colon ':' represents a HAS operator which is roughly
      synonymous with equality). {field} can refer to a proto or JSON field,
      or a synthetic field. Field names can be camelCase or snake_case.
    orderBy: Optional. Field to use to sort the list.
    pageSize: Optional. When requesting a page of resources, 'page_size'
      specifies number of resources to return. If unspecified, at most 500
      will be returned. The maximum value is 1000.
    pageToken: Optional. Token returned by previous call to
      'ListTerraformVersions' which specifies the position in the list from
      where to continue listing the resources.
    parent: Required. The parent in whose context the TerraformVersions are
      listed. The parent value is in the format:
      'projects/{project_id}/locations/{location}'.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)