from typing import TYPE_CHECKING, List, Optional, Sequence, Union
from urllib.parse import urlparse
from wandb.sdk.artifacts.storage_handler import StorageHandler
from wandb.sdk.lib.paths import FilePathStr, URIStr
def _get_handler(self, url: Union[FilePathStr, URIStr]) -> StorageHandler:
    parsed_url = urlparse(url)
    for handler in self._handlers:
        if handler.can_handle(parsed_url):
            return handler
    if self._default_handler is not None:
        return self._default_handler
    raise ValueError('No storage handler registered for url "%s"' % str(url))