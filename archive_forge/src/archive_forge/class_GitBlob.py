from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class GitBlob(CompletableGithubObject):
    """
    This class represents GitBlobs. The reference can be found here https://docs.github.com/en/rest/reference/git#blobs
    """

    def _initAttributes(self) -> None:
        self._content: Attribute[str] = NotSet
        self._encoding: Attribute[str] = NotSet
        self._sha: Attribute[str] = NotSet
        self._size: Attribute[int] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'sha': self._sha.value})

    @property
    def content(self) -> str:
        self._completeIfNotSet(self._content)
        return self._content.value

    @property
    def encoding(self) -> str:
        self._completeIfNotSet(self._encoding)
        return self._encoding.value

    @property
    def sha(self) -> str:
        self._completeIfNotSet(self._sha)
        return self._sha.value

    @property
    def size(self) -> int:
        self._completeIfNotSet(self._size)
        return self._size.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'content' in attributes:
            self._content = self._makeStringAttribute(attributes['content'])
        if 'encoding' in attributes:
            self._encoding = self._makeStringAttribute(attributes['encoding'])
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'size' in attributes:
            self._size = self._makeIntAttribute(attributes['size'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])