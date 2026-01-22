from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelEvaluationBiasConfig(_messages.Message):
    """Configuration for bias detection.

  Fields:
    biasSlices: Specification for how the data should be sliced for bias. It
      contains a list of slices, with limitation of two slices. The first
      slice of data will be the slice_a. The second slice in the list
      (slice_b) will be compared against the first slice. If only a single
      slice is provided, then slice_a will be compared against "not slice_a".
      Below are examples with feature "education" with value "low", "medium",
      "high" in the dataset: Example 1: bias_slices = [{'education': 'low'}] A
      single slice provided. In this case, slice_a is the collection of data
      with 'education' equals 'low', and slice_b is the collection of data
      with 'education' equals 'medium' or 'high'. Example 2: bias_slices =
      [{'education': 'low'}, {'education': 'high'}] Two slices provided. In
      this case, slice_a is the collection of data with 'education' equals
      'low', and slice_b is the collection of data with 'education' equals
      'high'.
    labels: Positive labels selection on the target field.
  """
    biasSlices = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluationSliceSliceSliceSpec', 1)
    labels = _messages.StringField(2, repeated=True)