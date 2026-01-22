import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def _create_repo(self, repo_id: str, private: Optional[bool]=None, token: Optional[Union[bool, str]]=None, repo_url: Optional[str]=None, organization: Optional[str]=None) -> str:
    """
        Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
        the token.
        """
    if repo_url is not None:
        warnings.warn('The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` instead.')
        if repo_id is not None:
            raise ValueError('`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`.')
        repo_id = repo_url.replace(f'{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/', '')
    if organization is not None:
        warnings.warn('The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`).')
        if not repo_id.startswith(organization):
            if '/' in repo_id:
                repo_id = repo_id.split('/')[-1]
            repo_id = f'{organization}/{repo_id}'
    url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
    return url.repo_id