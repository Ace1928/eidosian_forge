from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssetType(_messages.Message):
    """An asset type resource. An asset type defines the schema for the
  assets.It specifies configuration of all the fields present on the asset.

  Enums:
    MediaTypeValueValuesEnum: Specifies the kind of media held by assets of
      this asset type.

  Messages:
    AnnotationSetConfigsValue: Mapping of annotationSet ID to its
      configuration. The annotationSet ID will be used as the resource ID when
      GCMA creates the AnnotationSet internally. Detailed rules for a resource
      id are: 1. 1 character minimum, 63 characters maximum 2. only contains
      letters, digits, underscore and hyphen 3. starts with a letter if length
      == 1, starts with a letter or underscore if length > 1
    FacetConfigsValue: Mapping of facet name to its configuration. To update
      facets, use either "*" or "facet_configs" update mask.
    IndexedFieldConfigsValue: List of indexed fields (e.g.
      "metadata.file.url") to make available in searches with their
      corresponding properties.
    LabelsValue: The labels associated with this resource. Each label is a
      key-value pair.
    LinkConfigsValue: Mapping of a link field name to its configuration.
    LinkSetConfigsValue: Mapping of a collection of link sets to the set
      configuration.
    MetadataConfigsValue: Mapping of field name to its configuration.

  Fields:
    annotationSetConfigs: Mapping of annotationSet ID to its configuration.
      The annotationSet ID will be used as the resource ID when GCMA creates
      the AnnotationSet internally. Detailed rules for a resource id are: 1. 1
      character minimum, 63 characters maximum 2. only contains letters,
      digits, underscore and hyphen 3. starts with a letter if length == 1,
      starts with a letter or underscore if length > 1
    assetTypeStats: asset_type_stats stores stats on this asset type.
    createTime: Output only. The creation time.
    facetConfigs: Mapping of facet name to its configuration. To update
      facets, use either "*" or "facet_configs" update mask.
    featureConfigs: Configuration for IMS features, including languages for
      speech transcription.
    indexedFieldConfigs: List of indexed fields (e.g. "metadata.file.url") to
      make available in searches with their corresponding properties.
    labels: The labels associated with this resource. Each label is a key-
      value pair.
    linkConfigs: Mapping of a link field name to its configuration.
    linkSetConfigs: Mapping of a collection of link sets to the set
      configuration.
    mediaType: Specifies the kind of media held by assets of this asset type.
    metadataConfigs: Mapping of field name to its configuration.
    name: The resource name of the asset type, in the following form:
      `projects/{project}/locations/{location}/assetTypes/{type}`. Here {type}
      is a resource id. Detailed rules for a resource id are: 1. 1 character
      minimum, 63 characters maximum 2. only contains letters, digits,
      underscore and hyphen 3. starts with a letter if length == 1, starts
      with a letter or underscore if length > 1
    sortOrder: Specifies sort order for all assets of the type. If not
      specified, assets are sorted in reverse create_time order (newest
      first).
    updateTime: Output only. The last-modified time.
  """

    class MediaTypeValueValuesEnum(_messages.Enum):
        """Specifies the kind of media held by assets of this asset type.

    Values:
      MEDIA_TYPE_UNSPECIFIED: AssetTypes with unspecified media types hold
        generic assets.
      VIDEO: AssetTypes with video media types have the following properties:
        1. Have a required and immutable metadata field called 'video_file' of
        type 'system:gcs-file', which is the path to a video file. 2. Support
        searching the content of the provided video asset.
    """
        MEDIA_TYPE_UNSPECIFIED = 0
        VIDEO = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationSetConfigsValue(_messages.Message):
        """Mapping of annotationSet ID to its configuration. The annotationSet ID
    will be used as the resource ID when GCMA creates the AnnotationSet
    internally. Detailed rules for a resource id are: 1. 1 character minimum,
    63 characters maximum 2. only contains letters, digits, underscore and
    hyphen 3. starts with a letter if length == 1, starts with a letter or
    underscore if length > 1

    Messages:
      AdditionalProperty: An additional property for a
        AnnotationSetConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        AnnotationSetConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationSetConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A AnnotationSetConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('AnnotationSetConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FacetConfigsValue(_messages.Message):
        """Mapping of facet name to its configuration. To update facets, use
    either "*" or "facet_configs" update mask.

    Messages:
      AdditionalProperty: An additional property for a FacetConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type FacetConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FacetConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A FacetConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('FacetConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class IndexedFieldConfigsValue(_messages.Message):
        """List of indexed fields (e.g. "metadata.file.url") to make available in
    searches with their corresponding properties.

    Messages:
      AdditionalProperty: An additional property for a
        IndexedFieldConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        IndexedFieldConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a IndexedFieldConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A IndexedFieldConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('IndexedFieldConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels associated with this resource. Each label is a key-value
    pair.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LinkConfigsValue(_messages.Message):
        """Mapping of a link field name to its configuration.

    Messages:
      AdditionalProperty: An additional property for a LinkConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LinkConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LinkConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A LinkConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('LinkConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LinkSetConfigsValue(_messages.Message):
        """Mapping of a collection of link sets to the set configuration.

    Messages:
      AdditionalProperty: An additional property for a LinkSetConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LinkSetConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LinkSetConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A LinkSetConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('LinkSetConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataConfigsValue(_messages.Message):
        """Mapping of field name to its configuration.

    Messages:
      AdditionalProperty: An additional property for a MetadataConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MetadataConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A MetadataConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('MetadataConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotationSetConfigs = _messages.MessageField('AnnotationSetConfigsValue', 1)
    assetTypeStats = _messages.MessageField('AssetTypeStats', 2)
    createTime = _messages.StringField(3)
    facetConfigs = _messages.MessageField('FacetConfigsValue', 4)
    featureConfigs = _messages.MessageField('FeatureConfigs', 5)
    indexedFieldConfigs = _messages.MessageField('IndexedFieldConfigsValue', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    linkConfigs = _messages.MessageField('LinkConfigsValue', 8)
    linkSetConfigs = _messages.MessageField('LinkSetConfigsValue', 9)
    mediaType = _messages.EnumField('MediaTypeValueValuesEnum', 10)
    metadataConfigs = _messages.MessageField('MetadataConfigsValue', 11)
    name = _messages.StringField(12)
    sortOrder = _messages.MessageField('SortOrderConfig', 13)
    updateTime = _messages.StringField(14)