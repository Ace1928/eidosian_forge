import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import torch
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
from torch.serialization import MAP_LOCATION
def _check_repo_is_trusted(repo_owner, repo_name, owner_name_branch, trust_repo, calling_fn='load'):
    hub_dir = get_dir()
    filepath = os.path.join(hub_dir, 'trusted_list')
    if not os.path.exists(filepath):
        Path(filepath).touch()
    with open(filepath) as file:
        trusted_repos = tuple((line.strip() for line in file))
    trusted_repos_legacy = next(os.walk(hub_dir))[1]
    owner_name = '_'.join([repo_owner, repo_name])
    is_trusted = owner_name in trusted_repos or owner_name_branch in trusted_repos_legacy or repo_owner in _TRUSTED_REPO_OWNERS
    if trust_repo is None:
        if not is_trusted:
            warnings.warn(f"You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to {{calling_fn}}(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or {calling_fn}(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use {calling_fn}(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour")
        return
    if trust_repo is False or (trust_repo == 'check' and (not is_trusted)):
        response = input(f'The repository {owner_name} does not belong to the list of trusted repositories and as such cannot be downloaded. Do you trust this repository and wish to add it to the trusted list of repositories (y/N)?')
        if response.lower() in ('y', 'yes'):
            if is_trusted:
                print('The repository is already trusted.')
        elif response.lower() in ('n', 'no', ''):
            raise Exception('Untrusted repository.')
        else:
            raise ValueError(f'Unrecognized response {response}.')
    if not is_trusted:
        with open(filepath, 'a') as file:
            file.write(owner_name + '\n')