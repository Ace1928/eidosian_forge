from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceBundleSelector(_messages.Message):
    """ResourceBundleSelector determines what resource bundle and version to
  deploy.

  Fields:
    cloudBuildRepository: cloud_build_repository points to a gen 2 cloud build
      repository to use as the source of truth for KRM configs.
    resourceBundle: Required. resource_bundle refers to a resource bundle that
      is directly pushed by the user. Format:
      projects/{p}/locations/{l}/resourceBundles/{pkg}
    tag: Required. tag will support both the exact version as well as explicit
      tag. System will auto-generate tags which are useful such as tracking
      patch versions to support the concept of release channels. examples:
      v1.0.1 or v1.1.* or v1-stable
  """
    cloudBuildRepository = _messages.MessageField('CloudBuildRepository', 1)
    resourceBundle = _messages.StringField(2)
    tag = _messages.StringField(3)