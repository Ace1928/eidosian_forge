from typing import Any
from wandb.sdk.internal.internal_api import Api as InternalApi
def upload_file_retry(self, *args, **kwargs):
    return self.api.upload_file_retry(*args, **kwargs)