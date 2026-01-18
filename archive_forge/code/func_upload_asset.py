from __future__ import annotations
from datetime import datetime
from os.path import basename
from typing import Any, BinaryIO
import github.GitReleaseAsset
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts
def upload_asset(self, path: str, label: str='', content_type: Opt[str]=NotSet, name: Opt[str]=NotSet) -> github.GitReleaseAsset.GitReleaseAsset:
    """
        :calls: `POST https://<upload_url>/repos/{owner}/{repo}/releases/{release_id}/assets <https://docs.github.com/en/rest/reference/repos#upload-a-release-asset>`_
        """
    assert isinstance(path, str), path
    assert isinstance(label, str), label
    assert name is NotSet or isinstance(name, str), name
    post_parameters: dict[str, Any] = {'label': label}
    if name is NotSet:
        post_parameters['name'] = basename(path)
    else:
        post_parameters['name'] = name
    headers: dict[str, Any] = {}
    if content_type is not NotSet:
        headers['Content-Type'] = content_type
    resp_headers, data = self._requester.requestBlobAndCheck('POST', self.upload_url.split('{?')[0], parameters=post_parameters, headers=headers, input=path)
    return github.GitReleaseAsset.GitReleaseAsset(self._requester, resp_headers, data, completed=True)