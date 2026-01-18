from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
@validator('build_context_store')
@classmethod
def validate_build_context_store(cls, build_context_store: Optional[str]) -> Optional[str]:
    """Validate that the build context store is a valid container registry URI."""
    if build_context_store is None:
        return None
    for regex in [S3_URI_RE, GCS_URI_RE, AZURE_BLOB_REGEX]:
        if regex.match(build_context_store):
            return build_context_store
    raise ValueError('Invalid build context store. Build context store must be a URI for an S3 bucket, GCS bucket, or Azure blob.')