from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationConfig(_messages.Message):
    """User specified templates and configs for how to deidentify structured,
  unstructures, and image files. User must provide either a unstructured
  deidentify template or at least one redact image config.

  Fields:
    deidentifyTemplate: De-identify template. If this template is specified,
      it will serve as the default de-identify template. This template cannot
      contain `record_transformations` since it can be used for unstructured
      content such as free-form text files. If this template is not set, a
      default `ReplaceWithInfoTypeConfig` will be used to de-identify
      unstructured content.
    imageRedactTemplate: Image redact template. If this template is specified,
      it will serve as the de-identify template for images. If this template
      is not set, all findings in the image will be redacted with a black box.
    structuredDeidentifyTemplate: Structured de-identify template. If this
      template is specified, it will serve as the de-identify template for
      structured content such as delimited files and tables. If this template
      is not set but the `deidentify_template` is set, then
      `deidentify_template` will also apply to the structured content. If
      neither template is set, a default `ReplaceWithInfoTypeConfig` will be
      used to de-identify structured content.
  """
    deidentifyTemplate = _messages.StringField(1)
    imageRedactTemplate = _messages.StringField(2)
    structuredDeidentifyTemplate = _messages.StringField(3)