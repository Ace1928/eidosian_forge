from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ContainerTypeValueValuesEnum(_messages.Enum):
    """The container type of the QuotaInfo.

    Values:
      CONTAINER_TYPE_UNSPECIFIED: Unspecified container type.
      PROJECT: consumer project
      FOLDER: folder
      ORGANIZATION: organization
    """
    CONTAINER_TYPE_UNSPECIFIED = 0
    PROJECT = 1
    FOLDER = 2
    ORGANIZATION = 3