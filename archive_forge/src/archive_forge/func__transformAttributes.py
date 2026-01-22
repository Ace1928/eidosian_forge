from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def _transformAttributes(self, element: Dict[str, Any]) -> Dict[str, Any]:
    if self._attributesTransformer is None:
        return element
    return self._attributesTransformer(element)