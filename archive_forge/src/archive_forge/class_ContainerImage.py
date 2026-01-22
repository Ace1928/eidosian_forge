from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerImage(_messages.Message):
    """Definition of a container image for starting a notebook instance with
  the environment installed in a container.

  Fields:
    repository: Required. The path to the container image repository. For
      example: `gcr.io/{project_id}/{image_name}`
    tag: Optional. The tag of the container image. If not specified, this
      defaults to the latest tag.
  """
    repository = _messages.StringField(1)
    tag = _messages.StringField(2)