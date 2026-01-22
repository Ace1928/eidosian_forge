from __future__ import annotations
import logging
import warnings
from typing import Any, Callable
from lxml.html import HtmlElement
from extruct.dublincore import DublinCoreExtractor
from extruct.jsonld import JsonLdExtractor
from extruct.microformat import MicroformatExtractor
from extruct.opengraph import OpenGraphExtractor
from extruct.rdfa import RDFaExtractor
from extruct.uniform import _udublincore, _umicrodata_microformat, _uopengraph
from extruct.utils import parse_html, parse_xmldom_html
from extruct.w3cmicrodata import MicrodataExtractor

    htmlstring: string with valid html document;
    base_url: base url of the html document
    encoding: encoding of the html document
    syntaxes: list of syntaxes to extract, default SYNTAXES
    errors: set to 'log' to log the exceptions, 'ignore' to ignore them
            or 'strict'(default) to raise them
    uniform: if True uniform output format of all syntaxes to a list of dicts.
             Returned dicts structure:
             {'@context': 'http://example.com',
              '@type': 'example_type',
              /* All other the properties in keys here */
              }
    return_html_node: if True, it includes into the result a HTML node of
                      respective embedded metadata under 'htmlNode' key.
                      The feature is supported only by microdata syntax.
                      Each node is of `lxml.etree.Element` type.
    schema_context: schema's context for current page