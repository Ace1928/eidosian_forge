from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1EntityAnnotation(_messages.Message):
    """Set of detected entity features.

  Fields:
    boundingPoly: Image region to which this entity belongs. Not produced for
      `LABEL_DETECTION` features.
    confidence: **Deprecated. Use `score` instead.** The accuracy of the
      entity detection in an image. For example, for an image in which the
      "Eiffel Tower" entity is detected, this field represents the confidence
      that there is a tower in the query image. Range [0, 1].
    description: Entity textual description, expressed in its `locale`
      language.
    locale: The language code for the locale in which the entity textual
      `description` is expressed.
    locations: The location information for the detected entity. Multiple
      `LocationInfo` elements can be present because one location may indicate
      the location of the scene in the image, and another location may
      indicate the location of the place where the image was taken. Location
      information is usually present for landmarks.
    mid: Opaque entity ID. Some IDs may be available in [Google Knowledge
      Graph Search API](https://developers.google.com/knowledge-graph/).
    properties: Some entities may have optional user-supplied `Property`
      (name/value) fields, such a score or string that qualifies the entity.
    score: Overall score of the result. Range [0, 1].
    topicality: The relevancy of the ICA (Image Content Annotation) label to
      the image. For example, the relevancy of "tower" is likely higher to an
      image containing the detected "Eiffel Tower" than to an image containing
      a detected distant towering building, even though the confidence that
      there is a tower in each image may be the same. Range [0, 1].
  """
    boundingPoly = _messages.MessageField('GoogleCloudVisionV1p4beta1BoundingPoly', 1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    description = _messages.StringField(3)
    locale = _messages.StringField(4)
    locations = _messages.MessageField('GoogleCloudVisionV1p4beta1LocationInfo', 5, repeated=True)
    mid = _messages.StringField(6)
    properties = _messages.MessageField('GoogleCloudVisionV1p4beta1Property', 7, repeated=True)
    score = _messages.FloatField(8, variant=_messages.Variant.FLOAT)
    topicality = _messages.FloatField(9, variant=_messages.Variant.FLOAT)