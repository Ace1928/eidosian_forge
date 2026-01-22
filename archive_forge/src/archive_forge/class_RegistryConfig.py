from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
class RegistryConfig(BaseModel):
    """Configuration for registry block.

    Note that we don't forbid extra fields here because:
    - We want to allow all fields supported by each registry
    - We will perform validation on the registry object itself later
    - Registry block is being deprecated in favor of destination field in builder
    """
    type: Optional[RegistryType] = Field(None, description='The type of registry to use.')
    uri: Optional[str] = Field(None, description='The URI of the registry.')

    @validator('uri')
    @classmethod
    def validate_uri(cls, uri: str) -> str:
        for regex in [GCP_ARTIFACT_REGISTRY_URI_REGEX, AZURE_CONTAINER_REGISTRY_URI_REGEX, ELASTIC_CONTAINER_REGISTRY_URI_REGEX]:
            if regex.match(uri):
                return uri
        raise ValueError('Invalid uri. URI must be a repository URI for an ECR, ACR, or GCP Artifact Registry.')