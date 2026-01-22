from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelEvaluationSliceSliceSliceSpecSliceConfig(_messages.Message):
    """Specification message containing the config for this SliceSpec. When
  `kind` is selected as `value` and/or `range`, only a single slice will be
  computed. When `all_values` is present, a separate slice will be computed
  for each possible label/value for the corresponding key in `config`.
  Examples, with feature zip_code with values 12345, 23334, 88888 and feature
  country with values "US", "Canada", "Mexico" in the dataset: Example 1: {
  "zip_code": { "value": { "float_value": 12345.0 } } } A single slice for any
  data with zip_code 12345 in the dataset. Example 2: { "zip_code": { "range":
  { "low": 12345, "high": 20000 } } } A single slice containing data where the
  zip_codes between 12345 and 20000 For this example, data with the zip_code
  of 12345 will be in this slice. Example 3: { "zip_code": { "range": { "low":
  10000, "high": 20000 } }, "country": { "value": { "string_value": "US" } } }
  A single slice containing data where the zip_codes between 10000 and 20000
  has the country "US". For this example, data with the zip_code of 12345 and
  country "US" will be in this slice. Example 4: { "country": {"all_values": {
  "value": true } } } Three slices are computed, one for each unique country
  in the dataset. Example 5: { "country": { "all_values": { "value": true } },
  "zip_code": { "value": { "float_value": 12345.0 } } } Three slices are
  computed, one for each unique country in the dataset where the zip_code is
  also 12345. For this example, data with zip_code 12345 and country "US" will
  be in one slice, zip_code 12345 and country "Canada" in another slice, and
  zip_code 12345 and country "Mexico" in another slice, totaling 3 slices.

  Fields:
    allValues: If all_values is set to true, then all possible labels of the
      keyed feature will have another slice computed. Example:
      `{"all_values":{"value":true}}`
    range: A range of values for a numerical feature. Example:
      `{"range":{"low":10000.0,"high":50000.0}}` will capture 12345 and 23334
      in the slice.
    value: A unique specific value for a given feature. Example: `{ "value": {
      "string_value": "12345" } }`
  """
    allValues = _messages.BooleanField(1)
    range = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluationSliceSliceSliceSpecRange', 2)
    value = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluationSliceSliceSliceSpecValue', 3)