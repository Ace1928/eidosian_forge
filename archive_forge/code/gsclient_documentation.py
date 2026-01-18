from datetime import datetime
import mimetypes
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING, Tuple, Union
from ..client import Client, register_client_class
from ..cloudpath import implementation_registry
from ..enums import FileCacheMode
from .gspath import GSPath
Class constructor. Sets up a [`Storage
        Client`](https://googleapis.dev/python/storage/latest/client.html).
        Supports the following authentication methods of `Storage Client`.

        - Environment variable `"GOOGLE_APPLICATION_CREDENTIALS"` containing a
          path to a JSON credentials file for a Google service account. See
          [Authenticating as a Service
          Account](https://cloud.google.com/docs/authentication/production).
        - File path to a JSON credentials file for a Google service account.
        - OAuth2 Credentials object and a project name.
        - Instantiated and already authenticated `Storage Client`.

        If multiple methods are used, priority order is reverse of list above
        (later in list takes priority). If no authentication methods are used,
        then the client will be instantiated as anonymous, which will only have
        access to public buckets.

        Args:
            application_credentials (Optional[Union[str, os.PathLike]]): Path to Google service
                account credentials file.
            credentials (Optional[Credentials]): The OAuth2 Credentials to use for this client.
                See documentation for [`StorageClient`](
                https://googleapis.dev/python/storage/latest/client.html).
            project (Optional[str]): The project which the client acts on behalf of. See
                documentation for [`StorageClient`](
                https://googleapis.dev/python/storage/latest/client.html).
            storage_client (Optional[StorageClient]): Instantiated [`StorageClient`](
                https://googleapis.dev/python/storage/latest/client.html).
            file_cache_mode (Optional[Union[str, FileCacheMode]]): How often to clear the file cache; see
                [the caching docs](https://cloudpathlib.drivendata.org/stable/caching/) for more information
                about the options in cloudpathlib.eums.FileCacheMode.
            local_cache_dir (Optional[Union[str, os.PathLike]]): Path to directory to use as cache
                for downloaded files. If None, will use a temporary directory. Default can be set with
                the `CLOUDPATHLIB_LOCAL_CACHE_DIR` environment variable.
            content_type_method (Optional[Callable]): Function to call to guess media type (mimetype) when
                writing a file to the cloud. Defaults to `mimetypes.guess_type`. Must return a tuple (content type, content encoding).
        