from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExportDataConfig(_messages.Message):
    """Describes what part of the Dataset is to be exported, the destination of
  the export and how to export.

  Fields:
    annotationsFilter: An expression for filtering what part of the Dataset is
      to be exported. Only Annotations that match this filter will be
      exported. The filter syntax is the same as in ListAnnotations.
    fractionSplit: Split based on fractions defining the size of each set.
    gcsDestination: The Google Cloud Storage location where the output is to
      be written to. In the given directory a new directory will be created
      with name: `export-data--` where timestamp is in YYYY-MM-
      DDThh:mm:ss.sssZ ISO-8601 format. All export output will be written into
      that directory. Inside that directory, annotations with the same schema
      will be grouped into sub directories which are named with the
      corresponding annotations' schema title. Inside these sub directories, a
      schema.yaml will be created to describe the output format.
  """
    annotationsFilter = _messages.StringField(1)
    fractionSplit = _messages.MessageField('GoogleCloudAiplatformV1beta1ExportFractionSplit', 2)
    gcsDestination = _messages.MessageField('GoogleCloudAiplatformV1beta1GcsDestination', 3)