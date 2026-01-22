from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2Operation(_messages.Message):
    """Contains an operation for a resource loosely based on the JSON-PATCH
  format with support for: * Custom filters for describing partial array
  patch. * Extended path values for describing nested arrays. * Custom fields
  for describing the resource for which the operation is being described. *
  Allows extension to custom operations not natively supported by RFC6902. See
  https://tools.ietf.org/html/rfc6902 for details on the original RFC.

  Messages:
    PathFiltersValue: Set of filters to apply if `path` refers to array
      elements or nested array elements in order to narrow down to a single
      unique element that is being tested/modified. This is intended to be an
      exact match per filter. To perform advanced matching, use
      path_value_matchers. * Example: ``` { "/versions/*/name" : "it-123"
      "/versions/*/targetSize/percent": 20 } ``` * Example: ``` {
      "/bindings/*/role": "roles/owner" "/bindings/*/condition" : null } ``` *
      Example: ``` { "/bindings/*/role": "roles/owner" "/bindings/*/members/*"
      : ["x@example.com", "y@example.com"] } ``` When both path_filters and
      path_value_matchers are set, an implicit AND must be performed.
    PathValueMatchersValue: Similar to path_filters, this contains set of
      filters to apply if `path` field refers to array elements. This is meant
      to support value matching beyond exact match. To perform exact match,
      use path_filters. When both path_filters and path_value_matchers are
      set, an implicit AND must be performed.

  Fields:
    action: Type of this operation. Contains one of 'add', 'remove',
      'replace', 'move', 'copy', 'test' and 'custom' operations. This field is
      case-insensitive and always populated.
    customAction: Needed if action is 'custom'. The subtype of a custom
      action. i.e. ('navigate-to-page'). This field is also case-insensitive.
    path: Path to the target field being operated on. If the operation is at
      the resource level, then path should be "/". This field is always
      populated.
    pathFilters: Set of filters to apply if `path` refers to array elements or
      nested array elements in order to narrow down to a single unique element
      that is being tested/modified. This is intended to be an exact match per
      filter. To perform advanced matching, use path_value_matchers. *
      Example: ``` { "/versions/*/name" : "it-123"
      "/versions/*/targetSize/percent": 20 } ``` * Example: ``` {
      "/bindings/*/role": "roles/owner" "/bindings/*/condition" : null } ``` *
      Example: ``` { "/bindings/*/role": "roles/owner" "/bindings/*/members/*"
      : ["x@example.com", "y@example.com"] } ``` When both path_filters and
      path_value_matchers are set, an implicit AND must be performed.
    pathValueMatchers: Similar to path_filters, this contains set of filters
      to apply if `path` field refers to array elements. This is meant to
      support value matching beyond exact match. To perform exact match, use
      path_filters. When both path_filters and path_value_matchers are set, an
      implicit AND must be performed.
    resource: Contains the fully qualified resource name. This field is always
      populated. ex: //cloudresourcemanager.googleapis.com/projects/foo.
    resourceType: Type of GCP resource being modified/tested. This field is
      always populated. Example: cloudresourcemanager.googleapis.com/Project,
      compute.googleapis.com/Instance
    sourcePath: Can be set with action 'copy' or 'move' to indicate the source
      field within resource or source_resource, ignored if provided for other
      operation types.
    sourceResource: Can be set with action 'copy' to copy resource
      configuration across different resources of the same type. Example: A
      resource clone can be done via action = 'copy', path = "/", from = "/",
      source_resource = and resource_name = . This field is empty for all
      other values of `action`.
    value: Value for the `path` field. Will be set for
      actions:'add'/'replace'. Maybe set for action: 'test'. Either this or
      `value_matcher` will be set for 'test' operation. An exact match must be
      performed.
    valueMatcher: Can be set for action 'test' for advanced matching for the
      value of 'path' field. Either this or `value` will be set for 'test'
      operation.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PathFiltersValue(_messages.Message):
        """Set of filters to apply if `path` refers to array elements or nested
    array elements in order to narrow down to a single unique element that is
    being tested/modified. This is intended to be an exact match per filter.
    To perform advanced matching, use path_value_matchers. * Example: ``` {
    "/versions/*/name" : "it-123" "/versions/*/targetSize/percent": 20 } ``` *
    Example: ``` { "/bindings/*/role": "roles/owner" "/bindings/*/condition" :
    null } ``` * Example: ``` { "/bindings/*/role": "roles/owner"
    "/bindings/*/members/*" : ["x@example.com", "y@example.com"] } ``` When
    both path_filters and path_value_matchers are set, an implicit AND must be
    performed.

    Messages:
      AdditionalProperty: An additional property for a PathFiltersValue
        object.

    Fields:
      additionalProperties: Additional properties of type PathFiltersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PathFiltersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PathValueMatchersValue(_messages.Message):
        """Similar to path_filters, this contains set of filters to apply if
    `path` field refers to array elements. This is meant to support value
    matching beyond exact match. To perform exact match, use path_filters.
    When both path_filters and path_value_matchers are set, an implicit AND
    must be performed.

    Messages:
      AdditionalProperty: An additional property for a PathValueMatchersValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        PathValueMatchersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PathValueMatchersValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudRecommenderV1alpha2ValueMatcher attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudRecommenderV1alpha2ValueMatcher', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    action = _messages.StringField(1)
    customAction = _messages.StringField(2)
    path = _messages.StringField(3)
    pathFilters = _messages.MessageField('PathFiltersValue', 4)
    pathValueMatchers = _messages.MessageField('PathValueMatchersValue', 5)
    resource = _messages.StringField(6)
    resourceType = _messages.StringField(7)
    sourcePath = _messages.StringField(8)
    sourceResource = _messages.StringField(9)
    value = _messages.MessageField('extra_types.JsonValue', 10)
    valueMatcher = _messages.MessageField('GoogleCloudRecommenderV1alpha2ValueMatcher', 11)