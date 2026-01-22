from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PythonRepository(_messages.Message):
    """Configuration for a Python remote repository.

  Enums:
    PublicRepositoryValueValuesEnum: One of the publicly available Python
      repositories supported by Artifact Registry.

  Fields:
    artifactRegistryRepository: An Artifact Registry Repository.
    customRepository: Customer-specified remote repository.
    publicRepository: One of the publicly available Python repositories
      supported by Artifact Registry.
  """

    class PublicRepositoryValueValuesEnum(_messages.Enum):
        """One of the publicly available Python repositories supported by
    Artifact Registry.

    Values:
      PUBLIC_REPOSITORY_UNSPECIFIED: Unspecified repository.
      PYPI: PyPI.
    """
        PUBLIC_REPOSITORY_UNSPECIFIED = 0
        PYPI = 1
    artifactRegistryRepository = _messages.MessageField('GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigPythonRepositoryArtifactRegistryRepository', 1)
    customRepository = _messages.MessageField('GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigPythonRepositoryCustomRepository', 2)
    publicRepository = _messages.EnumField('PublicRepositoryValueValuesEnum', 3)