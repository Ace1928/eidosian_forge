from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class Download(CompletableGithubObject):
    """
    This class represents Downloads. The reference can be found here https://docs.github.com/en/rest/reference/repos
    """

    def _initAttributes(self) -> None:
        self._accesskeyid: Attribute[str] = NotSet
        self._acl: Attribute[str] = NotSet
        self._bucket: Attribute[str] = NotSet
        self._content_type: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._description: Attribute[str] = NotSet
        self._download_count: Attribute[int] = NotSet
        self._expirationdate: Attribute[datetime] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._id: Attribute[int] = NotSet
        self._mime_type: Attribute[str] = NotSet
        self._name: Attribute[str] = NotSet
        self._path: Attribute[str] = NotSet
        self._policy: Attribute[str] = NotSet
        self._prefix: Attribute[str] = NotSet
        self._redirect: Attribute[bool] = NotSet
        self._s3_url: Attribute[str] = NotSet
        self._signature: Attribute[str] = NotSet
        self._size: Attribute[int] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value})

    @property
    def accesskeyid(self) -> str:
        self._completeIfNotSet(self._accesskeyid)
        return self._accesskeyid.value

    @property
    def acl(self) -> str:
        self._completeIfNotSet(self._acl)
        return self._acl.value

    @property
    def bucket(self) -> str:
        self._completeIfNotSet(self._bucket)
        return self._bucket.value

    @property
    def content_type(self) -> str:
        self._completeIfNotSet(self._content_type)
        return self._content_type.value

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def description(self) -> str:
        self._completeIfNotSet(self._description)
        return self._description.value

    @property
    def download_count(self) -> int:
        self._completeIfNotSet(self._download_count)
        return self._download_count.value

    @property
    def expirationdate(self) -> datetime:
        self._completeIfNotSet(self._expirationdate)
        return self._expirationdate.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def id(self) -> int:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def mime_type(self) -> str:
        self._completeIfNotSet(self._mime_type)
        return self._mime_type.value

    @property
    def name(self) -> str:
        self._completeIfNotSet(self._name)
        return self._name.value

    @property
    def path(self) -> str:
        self._completeIfNotSet(self._path)
        return self._path.value

    @property
    def policy(self) -> str:
        self._completeIfNotSet(self._policy)
        return self._policy.value

    @property
    def prefix(self) -> str:
        self._completeIfNotSet(self._prefix)
        return self._prefix.value

    @property
    def redirect(self) -> bool:
        self._completeIfNotSet(self._redirect)
        return self._redirect.value

    @property
    def s3_url(self) -> str:
        self._completeIfNotSet(self._s3_url)
        return self._s3_url.value

    @property
    def signature(self) -> str:
        self._completeIfNotSet(self._signature)
        return self._signature.value

    @property
    def size(self) -> int:
        self._completeIfNotSet(self._size)
        return self._size.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    def delete(self) -> None:
        """
        :calls: `DELETE /repos/{owner}/{repo}/downloads/{id} <https://docs.github.com/en/rest/reference/repos>`_
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'accesskeyid' in attributes:
            self._accesskeyid = self._makeStringAttribute(attributes['accesskeyid'])
        if 'acl' in attributes:
            self._acl = self._makeStringAttribute(attributes['acl'])
        if 'bucket' in attributes:
            self._bucket = self._makeStringAttribute(attributes['bucket'])
        if 'content_type' in attributes:
            self._content_type = self._makeStringAttribute(attributes['content_type'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'description' in attributes:
            self._description = self._makeStringAttribute(attributes['description'])
        if 'download_count' in attributes:
            self._download_count = self._makeIntAttribute(attributes['download_count'])
        if 'expirationdate' in attributes:
            self._expirationdate = self._makeDatetimeAttribute(attributes['expirationdate'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'mime_type' in attributes:
            self._mime_type = self._makeStringAttribute(attributes['mime_type'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'path' in attributes:
            self._path = self._makeStringAttribute(attributes['path'])
        if 'policy' in attributes:
            self._policy = self._makeStringAttribute(attributes['policy'])
        if 'prefix' in attributes:
            self._prefix = self._makeStringAttribute(attributes['prefix'])
        if 'redirect' in attributes:
            self._redirect = self._makeBoolAttribute(attributes['redirect'])
        if 's3_url' in attributes:
            self._s3_url = self._makeStringAttribute(attributes['s3_url'])
        if 'signature' in attributes:
            self._signature = self._makeStringAttribute(attributes['signature'])
        if 'size' in attributes:
            self._size = self._makeIntAttribute(attributes['size'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])