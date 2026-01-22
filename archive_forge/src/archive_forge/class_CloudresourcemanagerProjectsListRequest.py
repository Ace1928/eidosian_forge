from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsListRequest(_messages.Message):
    """A CloudresourcemanagerProjectsListRequest object.

  Fields:
    filter: Optional. An expression for filtering the results of the request.
      Filter rules are case insensitive. If multiple fields are included in a
      filter query, the query will return results that match any of the
      fields. Some eligible fields for filtering are: + `name` + `id` +
      `labels.` (where *key* is the name of a label) + `parent.type` +
      `parent.id` + `lifecycleState` Some examples of filter queries: | Query
      | Description | |------------------|------------------------------------
      -----------------| | name:how* | The project's name starts with "how". |
      | name:Howl | The project's name is `Howl` or `howl`. | | name:HOWL |
      Equivalent to above. | | NAME:howl | Equivalent to above. | |
      labels.color:* | The project has the label `color`. | | labels.color:red
      | The project's label `color` has the value `red`. | | labels.color:red
      labels.size:big | The project's label `color` has the value `red` or its
      label `size` has the value `big`. | | lifecycleState:DELETE_REQUESTED |
      Only show projects that are pending deletion.| If no filter is
      specified, the call will return projects for which the user has the
      `resourcemanager.projects.get` permission. NOTE: To perform a by-parent
      query (eg., what projects are directly in a Folder), the caller must
      have the `resourcemanager.projects.list` permission on the parent and
      the filter must contain both a `parent.type` and a `parent.id`
      restriction (example: "parent.type:folder parent.id:123"). In this case
      an alternate search index is used which provides more consistent
      results.
    pageSize: Optional. The maximum number of Projects to return in the
      response. The server can return fewer Projects than requested. If
      unspecified, server picks an appropriate default.
    pageToken: Optional. A pagination token returned from a previous call to
      ListProjects that indicates from where listing should continue.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)