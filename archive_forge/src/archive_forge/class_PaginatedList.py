from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
class PaginatedList(PaginatedListBase[T]):
    """
    This class abstracts the `pagination of the API <https://docs.github.com/en/rest/guides/traversing-with-pagination>`_.

    You can simply enumerate through instances of this class::

        for repo in user.get_repos():
            print(repo.name)

    If you want to know the total number of items in the list::

        print(user.get_repos().totalCount)

    You can also index them or take slices::

        second_repo = user.get_repos()[1]
        first_repos = user.get_repos()[:10]

    If you want to iterate in reversed order, just do::

        for repo in user.get_repos().reversed:
            print(repo.name)

    And if you really need it, you can explicitly access a specific page::

        some_repos = user.get_repos().get_page(0)
        some_other_repos = user.get_repos().get_page(3)
    """

    def __init__(self, contentClass: Type[T], requester: Requester, firstUrl: str, firstParams: Any, headers: Optional[Dict[str, str]]=None, list_item: str='items', total_count_item: str='total_count', firstData: Optional[Any]=None, firstHeaders: Optional[Dict[str, Union[str, int]]]=None, attributesTransformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]=None):
        self.__requester = requester
        self.__contentClass = contentClass
        self.__firstUrl = firstUrl
        self.__firstParams = firstParams or ()
        self.__nextUrl = firstUrl
        self.__nextParams = firstParams or {}
        self.__headers = headers
        self.__list_item = list_item
        self.__total_count_item = total_count_item
        if self.__requester.per_page != 30:
            self.__nextParams['per_page'] = self.__requester.per_page
        self._reversed = False
        self.__totalCount: Optional[int] = None
        self._attributesTransformer = attributesTransformer
        first_page = []
        if firstData is not None and firstHeaders is not None:
            first_page = self._getPage(firstData, firstHeaders)
        super().__init__(first_page)

    def _transformAttributes(self, element: Dict[str, Any]) -> Dict[str, Any]:
        if self._attributesTransformer is None:
            return element
        return self._attributesTransformer(element)

    @property
    def totalCount(self) -> int:
        if not self.__totalCount:
            params = {} if self.__nextParams is None else self.__nextParams.copy()
            params.update({'per_page': 1})
            headers, data = self.__requester.requestJsonAndCheck('GET', self.__firstUrl, parameters=params, headers=self.__headers)
            if 'link' not in headers:
                if data and 'total_count' in data:
                    self.__totalCount = data['total_count']
                elif data:
                    if isinstance(data, dict):
                        data = data[self.__list_item]
                    self.__totalCount = len(data)
                else:
                    self.__totalCount = 0
            else:
                links = self.__parseLinkHeader(headers)
                lastUrl = links.get('last')
                if lastUrl:
                    self.__totalCount = int(parse_qs(lastUrl)['page'][0])
                else:
                    self.__totalCount = 0
        return self.__totalCount

    def _getLastPageUrl(self) -> Optional[str]:
        headers, data = self.__requester.requestJsonAndCheck('GET', self.__firstUrl, parameters=self.__nextParams, headers=self.__headers)
        links = self.__parseLinkHeader(headers)
        return links.get('last')

    @property
    def reversed(self) -> 'PaginatedList[T]':
        r = PaginatedList(self.__contentClass, self.__requester, self.__firstUrl, self.__firstParams, self.__headers, self.__list_item, attributesTransformer=self._attributesTransformer)
        r.__reverse()
        return r

    def __reverse(self) -> None:
        self._reversed = True
        lastUrl = self._getLastPageUrl()
        if lastUrl:
            self.__nextUrl = lastUrl

    def _couldGrow(self) -> bool:
        return self.__nextUrl is not None

    def _fetchNextPage(self) -> List[T]:
        headers, data = self.__requester.requestJsonAndCheck('GET', self.__nextUrl, parameters=self.__nextParams, headers=self.__headers)
        data = data if data else []
        return self._getPage(data, headers)

    def _getPage(self, data: Any, headers: Dict[str, Any]) -> List[T]:
        self.__nextUrl = None
        if len(data) > 0:
            links = self.__parseLinkHeader(headers)
            if self._reversed:
                if 'prev' in links:
                    self.__nextUrl = links['prev']
            elif 'next' in links:
                self.__nextUrl = links['next']
        self.__nextParams = None
        if self.__list_item in data:
            self.__totalCount = data.get(self.__total_count_item)
            data = data[self.__list_item]
        content = [self.__contentClass(self.__requester, headers, self._transformAttributes(element), completed=False) for element in data if element is not None]
        if self._reversed:
            return content[::-1]
        return content

    def __parseLinkHeader(self, headers: Dict[str, str]) -> Dict[str, str]:
        links = {}
        if 'link' in headers:
            linkHeaders = headers['link'].split(', ')
            for linkHeader in linkHeaders:
                url, rel, *rest = linkHeader.split('; ')
                url = url[1:-1]
                rel = rel[5:-1]
                links[rel] = url
        return links

    def get_page(self, page: int) -> List[T]:
        params = dict(self.__firstParams)
        if page != 0:
            params['page'] = page + 1
        if self.__requester.per_page != 30:
            params['per_page'] = self.__requester.per_page
        headers, data = self.__requester.requestJsonAndCheck('GET', self.__firstUrl, parameters=params, headers=self.__headers)
        if self.__list_item in data:
            self.__totalCount = data.get('total_count')
            data = data[self.__list_item]
        return [self.__contentClass(self.__requester, headers, self._transformAttributes(element), completed=False) for element in data]

    @classmethod
    def override_attributes(cls, overrides: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:

        def attributes_transformer(element: Dict[str, Any]) -> Dict[str, Any]:
            element = cls.merge_dicts(element, overrides)
            return element
        return attributes_transformer

    @classmethod
    def merge_dicts(cls, d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        d1 = d1.copy()
        for k, v in d2.items():
            if isinstance(v, dict):
                d1[k] = cls.merge_dicts(d1.get(k, {}), v)
            else:
                d1[k] = v
        return d1