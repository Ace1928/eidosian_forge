from typing import Any
from wandb.sdk.internal.internal_api import Api as InternalApi
def upload_multipart_file_chunk_retry(self, *args, **kwargs):
    return self.api.upload_multipart_file_chunk_retry(*args, **kwargs)