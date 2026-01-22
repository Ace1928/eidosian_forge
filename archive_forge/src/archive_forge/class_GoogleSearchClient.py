import abc
import time
import random
import asyncio
import contextlib
from pydantic import BaseModel
from lazyops.imports._aiohttpx import resolve_aiohttpx
from lazyops.imports._bs4 import resolve_bs4
from urllib.parse import quote_plus
import aiohttpx
from bs4 import BeautifulSoup, Tag
from .utils import filter_result, load_user_agents, load_cookie_jar, get_random_jitter
from typing import List, Optional, Dict, Any, Union, Tuple, Set, Callable, Awaitable, TypeVar, Generator, AsyncGenerator
class GoogleSearchClient(abc.ABC):

    def __init__(self, proxy: Optional[str]=None, user_agents: Optional[List[str]]=None, verify_ssl: Optional[bool]=True, default_lang: Optional[str]='en', timeout: Optional[int]=10, max_connections: Optional[int]=100, raise_exceptions: Optional[bool]=True, **kwargs):
        """
        Initializes the client
        """
        self.pre_init(**kwargs)
        self.proxy = proxy
        self.user_agents = user_agents or load_user_agents()
        self.verify_ssl = verify_ssl
        self.lang = default_lang
        self.raise_exceptions = raise_exceptions
        self.cookie_jar = load_cookie_jar()
        self.cookies = aiohttpx.Cookies(self.cookie_jar)
        self.timeout = timeout
        self.max_connections = max_connections
        self.api: aiohttpx.Client = None
        self.init_api(**kwargs)
        self.post_init(**kwargs)

    def pre_init(self, **kwargs):
        """
        Pre init
        """
        pass

    def post_init(self, **kwargs):
        """
        Post init
        """
        pass

    def init_api(self, **kwargs):
        """
        Initializes the api
        """
        if self.api is None:
            self.api = aiohttpx.Client(timeout=self.timeout, cookies=self.cookies, limits=aiohttpx.Limits(max_connections=self.max_connections), proxies={'all://': self.proxy} if self.proxy else None, follow_redirects=True)

    def _get_page(self, url: str, user_agent: Optional[str]=None, **kwargs) -> bytes:
        """
        Gets the page
        """
        user_agent = user_agent or random.choice(self.user_agents)
        request = self.api.build_request('GET', url)
        request.headers['User-Agent'] = user_agent
        response = self.api.send(request)
        if self.raise_exceptions:
            response.raise_for_status()
        self.cookies.extract_cookies(response)
        html = response.read()
        response.close()
        with contextlib.suppress(Exception):
            self.cookies.jar.save()
        return html

    def get_page(self, url: str, user_agent: Optional[str]=None, **kwargs) -> bytes:
        """
        Gets the page

        This is a wrapper that allows you to customize the 
        underlying logic of the get_page method
        """
        return self._get_page(url, user_agent, **kwargs)

    async def _aget_page(self, url: str, user_agent: Optional[str]=None, **kwargs) -> bytes:
        """
        Gets the page
        """
        user_agent = user_agent or random.choice(self.user_agents)
        request = await self.api.async_build_request('GET', url)
        request.headers['User-Agent'] = user_agent
        response = await self.api.async_send(request)
        if self.raise_exceptions:
            response.raise_for_status()
        self.cookies.extract_cookies(response)
        html = response.read()
        await response.aclose()
        with contextlib.suppress(Exception):
            self.cookies.jar.save()
        return html

    async def aget_page(self, url: str, user_agent: Optional[str]=None, **kwargs) -> bytes:
        """
        Gets the page

        This is a wrapper that allows you to customize the 
        underlying logic of the get_page method
        """
        return await self._aget_page(url, user_agent, **kwargs)

    def append_extra_params(self, url: str, extra_params: Dict[str, str]) -> str:
        """
        Appends extra params
        """
        if not extra_params:
            return url
        for k, v in extra_params.items():
            k = quote_plus(k)
            v = quote_plus(v)
            url = f'{url}&{k}={v}'
        return url

    def extract_anchors(self, html: Union[str, bytes, BeautifulSoup]) -> List[Tag]:
        """
        Extracts the anchors
        """
        soup = BeautifulSoup(html, 'html.parser') if isinstance(html, (bytes, str)) else html
        try:
            return soup.find(id='search').findAll('a')
        except AttributeError:
            gbar = soup.find(id='gbar')
            if gbar:
                gbar.clear()
            return soup.findAll('a')

    def extract_description(self, html: Union[str, bytes, BeautifulSoup]) -> List[Tag]:
        """
        Extracts the description
        """
        soup = BeautifulSoup(html, 'html.parser') if isinstance(html, (bytes, str)) else html
        try:
            return soup.find(id='search').findAll('span')
        except AttributeError:
            gbar = soup.find(id='gbar')
            if gbar:
                gbar.clear()
            return soup.findAll('span')

    def validate_extra_params(self, extra_params: Dict[str, str]):
        """
        Validates the extra params
        """
        for builtin_param in url_parameters:
            if builtin_param in extra_params:
                raise ValueError('GET parameter "%s" is overlapping with                     the built-in GET parameter', builtin_param)

    def _search(self, query: str, tld: Optional[str]='com', lang: Optional[str]='en', tbs: Optional[str]='0', safe: Optional[str]='off', num: Optional[int]=10, start: Optional[int]=0, stop: Optional[int]=None, pause: Optional[float]=2.0, country: Optional[str]='', include_title: Optional[bool]=False, include_description: Optional[bool]=False, extra_params: Optional[Dict[str, str]]=None, user_agent: Optional[str]=None, verify_ssl: Optional[bool]=True, **kwargs) -> Generator[RT, None, None]:
        """
        Searches the query
        """
        hashes = set()
        count = 0
        query = quote_plus(query)
        if not extra_params:
            extra_params = {}
        if extra_params:
            self.validate_extra_params(extra_params)
        sp = SearchParams(query=query, tld=tld, lang=lang, tbs=tbs, safe=safe, num=num, start=start, stop=stop, country=country, include_title=include_title, include_description=include_description)
        self.get_page(sp.url_home, user_agent, **kwargs)
        url = sp.start_url
        while not sp.stop or count < sp.stop:
            last_count = count
            url = self.append_extra_params(url, extra_params)
            time.sleep(get_random_jitter(pause))
            html = self.get_page(url, user_agent)
            soup = BeautifulSoup(html, 'html.parser')
            anchors = self.extract_anchors(soup)
            desc_anchors = self.extract_description(soup)
            for a, desc in zip(anchors, desc_anchors):
                result = sp.extract_result(a, desc, hashes)
                if not result:
                    continue
                yield result
                count += 1
                if sp.stop and count >= sp.stop:
                    return
            if last_count == count:
                break
            url = sp.get_next_url()

    async def _asearch(self, query: str, tld: Optional[str]='com', lang: Optional[str]='en', tbs: Optional[str]='0', safe: Optional[str]='off', num: Optional[int]=10, start: Optional[int]=0, stop: Optional[int]=None, pause: Optional[float]=2.0, country: Optional[str]='', include_title: Optional[bool]=False, include_description: Optional[bool]=False, extra_params: Optional[Dict[str, str]]=None, user_agent: Optional[str]=None, verify_ssl: Optional[bool]=True, **kwargs) -> AsyncGenerator[RT, None]:
        """
        Searches the query
        """
        hashes = set()
        count = 0
        query = quote_plus(query)
        if not extra_params:
            extra_params = {}
        if extra_params:
            self.validate_extra_params(extra_params)
        sp = SearchParams(query=query, tld=tld, lang=lang, tbs=tbs, safe=safe, num=num, start=start, stop=stop, country=country, include_title=include_title, include_description=include_description)
        await self.aget_page(sp.url_home, user_agent, **kwargs)
        url = sp.start_url
        while not sp.stop or count < sp.stop:
            last_count = count
            url = self.append_extra_params(url, extra_params)
            await asyncio.sleep(get_random_jitter(pause))
            html = await self.aget_page(url, user_agent, **kwargs)
            soup = BeautifulSoup(html, 'html.parser')
            anchors = self.extract_anchors(soup)
            desc_anchors = self.extract_description(soup)
            for a, desc in zip(anchors, desc_anchors):
                result = sp.extract_result(a, desc, hashes)
                if not result:
                    continue
                yield result
                count += 1
                if sp.stop and count >= sp.stop:
                    return
            if last_count == count:
                break
            url = sp.get_next_url()

    def search(self, query: str, tld: Optional[str]='com', lang: Optional[str]='en', tbs: Optional[str]='0', safe: Optional[str]='off', num: Optional[int]=10, start: Optional[int]=0, stop: Optional[int]=None, pause: Optional[float]=2.0, country: Optional[str]='', include_title: Optional[bool]=False, include_description: Optional[bool]=False, extra_params: Optional[Dict[str, str]]=None, user_agent: Optional[str]=None, verify_ssl: Optional[bool]=True, **kwargs) -> Generator[RT, None, None]:
        """
        Searches the query
        """
        yield from self._search(query=query, tld=tld, lang=lang, tbs=tbs, safe=safe, num=num, start=start, stop=stop, pause=pause, country=country, include_title=include_title, include_description=include_description, extra_params=extra_params, user_agent=user_agent, verify_ssl=verify_ssl, **kwargs)

    async def asearch(self, query: str, tld: Optional[str]='com', lang: Optional[str]='en', tbs: Optional[str]='0', safe: Optional[str]='off', num: Optional[int]=10, start: Optional[int]=0, stop: Optional[int]=None, pause: Optional[float]=2.0, country: Optional[str]='', include_title: Optional[bool]=False, include_description: Optional[bool]=False, extra_params: Optional[Dict[str, str]]=None, user_agent: Optional[str]=None, verify_ssl: Optional[bool]=True, **kwargs) -> AsyncGenerator[RT, None]:
        """
        Searches the query
        """
        async for item in self._asearch(query=query, tld=tld, lang=lang, tbs=tbs, safe=safe, num=num, start=start, stop=stop, pause=pause, country=country, include_title=include_title, include_description=include_description, extra_params=extra_params, user_agent=user_agent, verify_ssl=verify_ssl, **kwargs):
            yield item