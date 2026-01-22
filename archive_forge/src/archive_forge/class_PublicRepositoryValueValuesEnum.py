from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicRepositoryValueValuesEnum(_messages.Enum):
    """One of the publicly available Python repositories supported by
    Artifact Registry.

    Values:
      PUBLIC_REPOSITORY_UNSPECIFIED: Unspecified repository.
      PYPI: PyPI.
    """
    PUBLIC_REPOSITORY_UNSPECIFIED = 0
    PYPI = 1