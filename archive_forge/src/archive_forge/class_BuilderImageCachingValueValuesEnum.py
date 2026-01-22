from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuilderImageCachingValueValuesEnum(_messages.Enum):
    """Immutable. Controls how the worker pool caches images. If unspecified
    during worker pool creation, this field is defaulted to CACHING_DISABLED.

    Values:
      BUILDER_IMAGE_CACHING_UNSPECIFIED: Default enum type. This should not be
        used.
      CACHING_DISABLED: DinD caching is disabled and no caching resources are
        provisioned.
      VOLUME_CACHING: A PersistentVolumeClaim is provisioned for caching.
    """
    BUILDER_IMAGE_CACHING_UNSPECIFIED = 0
    CACHING_DISABLED = 1
    VOLUME_CACHING = 2