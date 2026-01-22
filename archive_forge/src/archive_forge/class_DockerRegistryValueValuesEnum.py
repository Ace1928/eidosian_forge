from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DockerRegistryValueValuesEnum(_messages.Enum):
    """Docker Registry to use for this deployment. This configuration is only
    applicable to 1st Gen functions, 2nd Gen functions can only use Artifact
    Registry. If unspecified, it defaults to `ARTIFACT_REGISTRY`. If
    `docker_repository` field is specified, this field should either be left
    unspecified or set to `ARTIFACT_REGISTRY`.

    Values:
      DOCKER_REGISTRY_UNSPECIFIED: Unspecified.
      CONTAINER_REGISTRY: Docker images will be stored in multi-regional
        Container Registry repositories named `gcf`.
      ARTIFACT_REGISTRY: Docker images will be stored in regional Artifact
        Registry repositories. By default, GCF will create and use
        repositories named `gcf-artifacts` in every region in which a function
        is deployed. But the repository to use can also be specified by the
        user using the `docker_repository` field.
    """
    DOCKER_REGISTRY_UNSPECIFIED = 0
    CONTAINER_REGISTRY = 1
    ARTIFACT_REGISTRY = 2