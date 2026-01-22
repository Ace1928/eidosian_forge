import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
class CompletableGithubObject(GithubObject):

    def __init__(self, requester: 'Requester', headers: Dict[str, Union[str, int]], attributes: Dict[str, Any], completed: bool):
        super().__init__(requester, headers, attributes, completed)
        self.__completed = completed

    def __eq__(self, other: Any) -> bool:
        return other.__class__ is self.__class__ and other._url.value == self._url.value

    def __hash__(self) -> int:
        return hash(self._url.value)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def _completeIfNotSet(self, value: Attribute) -> None:
        if isinstance(value, _NotSetType):
            self._completeIfNeeded()

    def _completeIfNeeded(self) -> None:
        if not self.__completed:
            self.__complete()

    def __complete(self) -> None:
        if self._url.value is None:
            raise IncompletableObject(400, message='Returned object contains no URL')
        headers, data = self._requester.requestJsonAndCheck('GET', self._url.value)
        self._storeAndUseAttributes(headers, data)
        self.__completed = True

    def update(self, additional_headers: Optional[Dict[str, Any]]=None) -> bool:
        """
        Check and update the object with conditional request
        :rtype: Boolean value indicating whether the object is changed
        """
        conditionalRequestHeader = dict()
        if self.etag is not None:
            conditionalRequestHeader[Consts.REQ_IF_NONE_MATCH] = self.etag
        if self.last_modified is not None:
            conditionalRequestHeader[Consts.REQ_IF_MODIFIED_SINCE] = self.last_modified
        if additional_headers is not None:
            conditionalRequestHeader.update(additional_headers)
        status, responseHeaders, output = self._requester.requestJson('GET', self._url.value, headers=conditionalRequestHeader)
        if status == 304:
            return False
        else:
            headers, data = self._requester._Requester__check(status, responseHeaders, output)
            self._storeAndUseAttributes(headers, data)
            self.__completed = True
            return True