from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class SourceImport(CompletableGithubObject):
    """
    This class represents SourceImports. The reference can be found here https://docs.github.com/en/rest/reference/migrations#source-imports
    """

    def _initAttributes(self) -> None:
        self._authors_count: Attribute[int] = NotSet
        self._authors_url: Attribute[str] = NotSet
        self._has_large_files: Attribute[bool] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._large_files_count: Attribute[int] = NotSet
        self._large_files_size: Attribute[int] = NotSet
        self._repository_url: Attribute[str] = NotSet
        self._status: Attribute[str] = NotSet
        self._status_text: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet
        self._use_lfs: Attribute[str] = NotSet
        self._vcs: Attribute[str] = NotSet
        self._vcs_url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'vcs_url': self._vcs_url.value, 'repository_url': self._repository_url.value, 'status': self._status.value, 'url': self._url.value})

    @property
    def authors_count(self) -> int:
        self._completeIfNotSet(self._authors_count)
        return self._authors_count.value

    @property
    def authors_url(self) -> str:
        self._completeIfNotSet(self._authors_url)
        return self._authors_url.value

    @property
    def has_large_files(self) -> bool:
        self._completeIfNotSet(self._has_large_files)
        return self._has_large_files.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def large_files_count(self) -> int:
        self._completeIfNotSet(self._large_files_count)
        return self._large_files_count.value

    @property
    def large_files_size(self) -> int:
        self._completeIfNotSet(self._large_files_size)
        return self._large_files_size.value

    @property
    def repository_url(self) -> str:
        self._completeIfNotSet(self._repository_url)
        return self._repository_url.value

    @property
    def status(self) -> str:
        self._completeIfNotSet(self._status)
        return self._status.value

    @property
    def status_text(self) -> str:
        self._completeIfNotSet(self._status_text)
        return self._status_text.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def use_lfs(self) -> str:
        self._completeIfNotSet(self._use_lfs)
        return self._use_lfs.value

    @property
    def vcs(self) -> str:
        self._completeIfNotSet(self._vcs)
        return self._vcs.value

    @property
    def vcs_url(self) -> str:
        self._completeIfNotSet(self._vcs_url)
        return self._vcs_url.value

    def update(self, additional_headers: None | dict[str, Any]=None) -> bool:
        import_header = {'Accept': Consts.mediaTypeImportPreview}
        return super().update(additional_headers=import_header)

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'authors_count' in attributes:
            self._authors_count = self._makeIntAttribute(attributes['authors_count'])
        if 'authors_url' in attributes:
            self._authors_url = self._makeStringAttribute(attributes['authors_url'])
        if 'has_large_files' in attributes:
            self._has_large_files = self._makeBoolAttribute(attributes['has_large_files'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'large_files_count' in attributes:
            self._large_files_count = self._makeIntAttribute(attributes['large_files_count'])
        if 'large_files_size' in attributes:
            self._large_files_size = self._makeIntAttribute(attributes['large_files_size'])
        if 'repository_url' in attributes:
            self._repository_url = self._makeStringAttribute(attributes['repository_url'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])
        if 'status_text' in attributes:
            self._status_text = self._makeStringAttribute(attributes['status_text'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'use_lfs' in attributes:
            self._use_lfs = self._makeStringAttribute(attributes['use_lfs'])
        if 'vcs' in attributes:
            self._vcs = self._makeStringAttribute(attributes['vcs'])
        if 'vcs_url' in attributes:
            self._vcs_url = self._makeStringAttribute(attributes['vcs_url'])