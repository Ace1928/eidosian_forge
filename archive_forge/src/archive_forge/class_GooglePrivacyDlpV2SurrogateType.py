from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2SurrogateType(_messages.Message):
    """Message for detecting output from deidentification transformations such
  as [`CryptoReplaceFfxFpeConfig`](https://cloud.google.com/sensitive-data-pro
  tection/docs/reference/rest/v2/organizations.deidentifyTemplates#cryptorepla
  ceffxfpeconfig). These types of transformations are those that perform
  pseudonymization, thereby producing a "surrogate" as output. This should be
  used in conjunction with a field on the transformation such as
  `surrogate_info_type`. This CustomInfoType does not support the use of
  `detection_rules`.
  """