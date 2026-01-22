import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Union, cast
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class AsyncHtmlLoader(BaseLoader):
    """Load `HTML` asynchronously."""

    def __init__(self, web_path: Union[str, List[str]], header_template: Optional[dict]=None, verify_ssl: Optional[bool]=True, proxies: Optional[dict]=None, autoset_encoding: bool=True, encoding: Optional[str]=None, default_parser: str='html.parser', requests_per_second: int=2, requests_kwargs: Optional[Dict[str, Any]]=None, raise_for_status: bool=False, ignore_load_errors: bool=False):
        """Initialize with a webpage path."""
        if isinstance(web_path, str):
            self.web_paths = [web_path]
        elif isinstance(web_path, List):
            self.web_paths = web_path
        headers = header_template or default_header_template
        if not headers.get('User-Agent'):
            try:
                from fake_useragent import UserAgent
                headers['User-Agent'] = UserAgent().random
            except ImportError:
                logger.info('fake_useragent not found, using default user agent.To get a realistic header for requests, `pip install fake_useragent`.')
        self.session = requests.Session()
        self.session.headers = dict(headers)
        self.session.verify = verify_ssl
        if proxies:
            self.session.proxies.update(proxies)
        self.requests_per_second = requests_per_second
        self.default_parser = default_parser
        self.requests_kwargs = requests_kwargs or {}
        self.raise_for_status = raise_for_status
        self.autoset_encoding = autoset_encoding
        self.encoding = encoding
        self.ignore_load_errors = ignore_load_errors

    def _fetch_valid_connection_docs(self, url: str) -> Any:
        if self.ignore_load_errors:
            try:
                return self.session.get(url, **self.requests_kwargs)
            except Exception as e:
                warnings.warn(str(e))
                return None
        return self.session.get(url, **self.requests_kwargs)

    @staticmethod
    def _check_parser(parser: str) -> None:
        """Check that parser is valid for bs4."""
        valid_parsers = ['html.parser', 'lxml', 'xml', 'lxml-xml', 'html5lib']
        if parser not in valid_parsers:
            raise ValueError('`parser` must be one of ' + ', '.join(valid_parsers) + '.')

    def _scrape(self, url: str, parser: Union[str, None]=None, bs_kwargs: Optional[dict]=None) -> Any:
        from bs4 import BeautifulSoup
        if parser is None:
            if url.endswith('.xml'):
                parser = 'xml'
            else:
                parser = self.default_parser
        self._check_parser(parser)
        html_doc = self._fetch_valid_connection_docs(url)
        if not getattr(html_doc, 'ok', False):
            return None
        if self.raise_for_status:
            html_doc.raise_for_status()
        if self.encoding is not None:
            html_doc.encoding = self.encoding
        elif self.autoset_encoding:
            html_doc.encoding = html_doc.apparent_encoding
        return BeautifulSoup(html_doc.text, parser, **bs_kwargs or {})

    async def _fetch(self, url: str, retries: int=3, cooldown: int=2, backoff: float=1.5) -> str:
        async with aiohttp.ClientSession() as session:
            for i in range(retries):
                try:
                    async with session.get(url, headers=self.session.headers, ssl=None if self.session.verify else False) as response:
                        try:
                            text = await response.text()
                        except UnicodeDecodeError:
                            logger.error(f'Failed to decode content from {url}')
                            text = ''
                        return text
                except aiohttp.ClientConnectionError as e:
                    if i == retries - 1 and self.ignore_load_errors:
                        logger.warning(f'Error fetching {url} after {retries} retries.')
                        return ''
                    elif i == retries - 1:
                        raise
                    else:
                        logger.warning(f'Error fetching {url} with attempt {i + 1}/{retries}: {e}. Retrying...')
                        await asyncio.sleep(cooldown * backoff ** i)
        raise ValueError('retry count exceeded')

    async def _fetch_with_rate_limit(self, url: str, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            return await self._fetch(url)

    async def fetch_all(self, urls: List[str]) -> Any:
        """Fetch all urls concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            from tqdm.asyncio import tqdm_asyncio
            return await tqdm_asyncio.gather(*tasks, desc='Fetching pages', ascii=True, mininterval=1)
        except ImportError:
            warnings.warn('For better logging of progress, `pip install tqdm`')
            return await asyncio.gather(*tasks)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for doc in self.load():
            yield doc

    def load(self) -> List[Document]:
        """Load text from the url(s) in web_path."""
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.fetch_all(self.web_paths))
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self.fetch_all(self.web_paths))
        docs = []
        for i, text in enumerate(cast(List[str], results)):
            soup = self._scrape(self.web_paths[i])
            if not soup:
                continue
            metadata = _build_metadata(soup, self.web_paths[i])
            docs.append(Document(page_content=text, metadata=metadata))
        return docs