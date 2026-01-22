from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildNote(_messages.Message):
    """Note holding the version of the provider's builder and the signature of
  the provenance message in the build details occurrence.

  Fields:
    builderVersion: Required. Immutable. Version of the builder which produced
      this build.
  """
    builderVersion = _messages.StringField(1)