from ._base import *
from datetime import datetime
from typing import Optional, List, Union
from dataclasses import dataclass
from lazyops.lazyclasses.api import lazyclass
from lazyops.utils import timed_cache
@lazyclass
@dataclass
class GDELTArticle:
    url: str
    url_mobile: str = ''
    title: str = ''
    seendate: str = ''
    socialimage: str = ''
    domain: str = ''
    language: str = ''
    sourcecountry: str = ''
    text: Optional[str] = None
    extraction: Optional[Article] = None

    def parse(self):
        if self.extraction is not None:
            return
        self.extraction = Article(url=self.url)
        self.extraction._run()
        if self.extraction.extracted:
            self.text = self.extraction.text

    async def async_parse(self):
        if self.extraction is not None:
            return
        self.extraction = Article(url=self.url)
        self.extraction._run()
        if self.extraction.extracted:
            self.text = self.extraction.text