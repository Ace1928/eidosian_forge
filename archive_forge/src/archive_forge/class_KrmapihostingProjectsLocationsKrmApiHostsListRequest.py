from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KrmapihostingProjectsLocationsKrmApiHostsListRequest(_messages.Message):
    """A KrmapihostingProjectsLocationsKrmApiHostsListRequest object.

  Fields:
    filter: Lists the KrmApiHosts that match the filter expression. A filter
      expression filters the resources listed in the response. The expression
      must be of the form '{field} {operator} {value}' where operators: '<',
      '>', '<=', '>=', '!=', '=', ':' are supported (colon ':' represents a
      HAS operator which is roughly synonymous with equality). {field} can
      refer to a proto or JSON field, or a synthetic field. Field names can be
      camelCase or snake_case. Examples: - Filter by name: name =
      "projects/foo-proj/locations/us-central1/krmApiHosts/bar - Filter by
      labels: - Resources that have a key called 'foo' labels.foo:* -
      Resources that have a key called 'foo' whose value is 'bar' labels.foo =
      bar - Filter by state: - Members in CREATING state. state = CREATING
    orderBy: Field to use to sort the list.
    pageSize: When requesting a 'page' of resources, 'page_size' specifies
      number of resources to return. If unspecified or set to 0, all resources
      will be returned.
    pageToken: Token returned by previous call to 'ListKrmApiHosts' which
      specifies the position in the list from where to continue listing the
      resources.
    parent: Required. The parent in whose context the KrmApiHosts are listed.
      The parent value is in the format:
      'projects/{project_id}/locations/{location}'.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)