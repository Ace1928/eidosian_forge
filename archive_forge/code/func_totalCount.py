from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
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