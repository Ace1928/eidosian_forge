import json
import os
import tempfile
import time
import urllib
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.lib import ipython, json_util, runid
from wandb.sdk.lib.paths import LogicalPath
@normalize_exceptions
def upload_file(self, path, root='.'):
    """Upload a file.

        Arguments:
            path (str): name of file to upload.
            root (str): the root path to save the file relative to.  i.e.
                If you want to have the file saved in the run as "my_dir/file.txt"
                and you're currently in "my_dir" you would set root to "../".

        Returns:
            A `File` matching the name argument.
        """
    api = InternalApi(default_settings={'entity': self.entity, 'project': self.project}, retry_timedelta=RETRY_TIMEDELTA)
    api.set_current_run_id(self.id)
    root = os.path.abspath(root)
    name = os.path.relpath(path, root)
    with open(os.path.join(root, name), 'rb') as f:
        api.push({LogicalPath(name): f})
    return public.Files(self.client, self, [name])[0]