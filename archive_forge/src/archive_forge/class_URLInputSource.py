from __future__ import annotations
import codecs
import os
import pathlib
import sys
from io import BufferedIOBase, BytesIO, RawIOBase, StringIO, TextIOBase, TextIOWrapper
from typing import (
from urllib.parse import urljoin
from urllib.request import Request, url2pathname
from xml.sax import xmlreader
import rdflib.util
from rdflib import __version__
from rdflib._networking import _urlopen
from rdflib.namespace import Namespace
from rdflib.term import URIRef
class URLInputSource(InputSource):
    """
    Constructs an RDFLib Parser InputSource from a URL to read it from the Web.
    """
    links: List[str]

    @classmethod
    def getallmatchingheaders(cls, message: 'Message', name) -> List[str]:
        name = name.lower()
        return [val for key, val in message.items() if key.lower() == name]

    @classmethod
    def get_links(cls, response: addinfourl) -> List[str]:
        linkslines = cls.getallmatchingheaders(response.headers, 'Link')
        retarray: List[str] = []
        for linksline in linkslines:
            links = [linkstr.strip() for linkstr in linksline.split(',')]
            for link in links:
                retarray.append(link)
        return retarray

    def get_alternates(self, type_: Optional[str]=None) -> List[str]:
        typestr: Optional[str] = f'type="{type_}"' if type_ else None
        relstr = 'rel="alternate"'
        alts = []
        for link in self.links:
            parts = [p.strip() for p in link.split(';')]
            if relstr not in parts:
                continue
            if typestr:
                if typestr in parts:
                    alts.append(parts[0].strip('<>'))
            else:
                alts.append(parts[0].strip('<>'))
        return alts

    def __init__(self, system_id: Optional[str]=None, format: Optional[str]=None):
        super(URLInputSource, self).__init__(system_id)
        self.url = system_id
        myheaders = dict(headers)
        if format == 'xml':
            myheaders['Accept'] = 'application/rdf+xml, */*;q=0.1'
        elif format == 'n3':
            myheaders['Accept'] = 'text/n3, */*;q=0.1'
        elif format in ['turtle', 'ttl']:
            myheaders['Accept'] = 'text/turtle, application/x-turtle, */*;q=0.1'
        elif format == 'nt':
            myheaders['Accept'] = 'text/plain, */*;q=0.1'
        elif format == 'trig':
            myheaders['Accept'] = 'application/trig, */*;q=0.1'
        elif format == 'trix':
            myheaders['Accept'] = 'application/trix, */*;q=0.1'
        elif format == 'json-ld':
            myheaders['Accept'] = 'application/ld+json, application/json;q=0.9, */*;q=0.1'
        else:
            from rdflib.parser import Parser
            from rdflib.plugin import plugins
            acc = []
            for p in plugins(kind=Parser):
                if '/' in p.name:
                    acc.append(p.name)
            myheaders['Accept'] = ', '.join(acc)
        req = Request(system_id, None, myheaders)
        response: addinfourl = _urlopen(req)
        self.url = response.geturl()
        self.links = self.get_links(response)
        if format in ('json-ld', 'application/ld+json'):
            alts = self.get_alternates(type_='application/ld+json')
            for link in alts:
                full_link = urljoin(self.url, link)
                if full_link != self.url and full_link != system_id:
                    response = _urlopen(Request(full_link))
                    self.url = response.geturl()
                    break
        self.setPublicId(self.url)
        content_types = self.getallmatchingheaders(response.headers, 'content-type')
        self.content_type = content_types[0] if content_types else None
        if self.content_type is not None:
            self.content_type = self.content_type.split(';', 1)[0]
        self.setByteStream(response)
        self.response_info = response.info()

    def __repr__(self) -> str:
        return self.url