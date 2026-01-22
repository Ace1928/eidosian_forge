from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeepExtensionsConfig(_messages.Message):
    """The behavior for handling FHIR extensions that aren't otherwise
  specified for de-identification. If provided, all extensions are preserved
  during de-identification by default. If unspecified, all extensions are
  removed during de-identification by default.
  """