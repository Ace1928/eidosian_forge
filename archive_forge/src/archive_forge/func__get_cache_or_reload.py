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
def _get_cache_or_reload(github, force_reload, trust_repo, calling_fn, verbose=True, skip_validation=False):
    hub_dir = get_dir()
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)
    repo_owner, repo_name, ref = _parse_repo_info(github)
    normalized_br = ref.replace('/', '_')
    owner_name_branch = '_'.join([repo_owner, repo_name, normalized_br])
    repo_dir = os.path.join(hub_dir, owner_name_branch)
    _check_repo_is_trusted(repo_owner, repo_name, owner_name_branch, trust_repo=trust_repo, calling_fn=calling_fn)
    use_cache = not force_reload and os.path.exists(repo_dir)
    if use_cache:
        if verbose:
            sys.stderr.write(f'Using cache found in {repo_dir}\n')
    else:
        if not skip_validation:
            _validate_not_a_forked_repo(repo_owner, repo_name, ref)
        cached_file = os.path.join(hub_dir, normalized_br + '.zip')
        _remove_if_exists(cached_file)
        try:
            url = _git_archive_link(repo_owner, repo_name, ref)
            sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
            download_url_to_file(url, cached_file, progress=False)
        except HTTPError as err:
            if err.code == 300:
                warnings.warn(f"The ref {ref} is ambiguous. Perhaps it is both a tag and a branch in the repo? Torchhub will now assume that it's a branch. You can disambiguate tags and branches by explicitly passing refs/heads/branch_name or refs/tags/tag_name as the ref. That might require using skip_validation=True.")
                disambiguated_branch_ref = f'refs/heads/{ref}'
                url = _git_archive_link(repo_owner, repo_name, ref=disambiguated_branch_ref)
                download_url_to_file(url, cached_file, progress=False)
            else:
                raise
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            cached_zipfile.extractall(hub_dir)
        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)
    return repo_dir