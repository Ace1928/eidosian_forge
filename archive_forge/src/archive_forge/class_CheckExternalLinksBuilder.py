import json
import re
import socket
import time
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from os import path
from queue import PriorityQueue, Queue
from threading import Thread
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, cast
from urllib.parse import unquote, urlparse, urlunparse
from docutils import nodes
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, TooManyRedirects
from sphinx.application import Sphinx
from sphinx.builders.dummy import DummyBuilder
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import encode_uri, logging, requests
from sphinx.util.console import darkgray, darkgreen, purple, red, turquoise  # type: ignore
from sphinx.util.nodes import get_node_line
class CheckExternalLinksBuilder(DummyBuilder):
    """
    Checks for broken external links.
    """
    name = 'linkcheck'
    epilog = __('Look for any errors in the above output or in %(outdir)s/output.txt')

    def init(self) -> None:
        self.broken_hyperlinks = 0
        self.hyperlinks: Dict[str, Hyperlink] = {}
        socket.setdefaulttimeout(5.0)

    def process_result(self, result: CheckResult) -> None:
        filename = self.env.doc2path(result.docname, False)
        linkstat = {'filename': filename, 'lineno': result.lineno, 'status': result.status, 'code': result.code, 'uri': result.uri, 'info': result.message}
        self.write_linkstat(linkstat)
        if result.status == 'unchecked':
            return
        if result.status == 'working' and result.message == 'old':
            return
        if result.lineno:
            logger.info('(%16s: line %4d) ', result.docname, result.lineno, nonl=True)
        if result.status == 'ignored':
            if result.message:
                logger.info(darkgray('-ignored- ') + result.uri + ': ' + result.message)
            else:
                logger.info(darkgray('-ignored- ') + result.uri)
        elif result.status == 'local':
            logger.info(darkgray('-local-   ') + result.uri)
            self.write_entry('local', result.docname, filename, result.lineno, result.uri)
        elif result.status == 'working':
            logger.info(darkgreen('ok        ') + result.uri + result.message)
        elif result.status == 'broken':
            if self.app.quiet or self.app.warningiserror:
                logger.warning(__('broken link: %s (%s)'), result.uri, result.message, location=(result.docname, result.lineno))
            else:
                logger.info(red('broken    ') + result.uri + red(' - ' + result.message))
            self.write_entry('broken', result.docname, filename, result.lineno, result.uri + ': ' + result.message)
            self.broken_hyperlinks += 1
        elif result.status == 'redirected':
            try:
                text, color = {301: ('permanently', purple), 302: ('with Found', purple), 303: ('with See Other', purple), 307: ('temporarily', turquoise), 308: ('permanently', purple)}[result.code]
            except KeyError:
                text, color = ('with unknown code', purple)
            linkstat['text'] = text
            if self.config.linkcheck_allowed_redirects:
                logger.warning('redirect  ' + result.uri + ' - ' + text + ' to ' + result.message, location=(result.docname, result.lineno))
            else:
                logger.info(color('redirect  ') + result.uri + color(' - ' + text + ' to ' + result.message))
            self.write_entry('redirected ' + text, result.docname, filename, result.lineno, result.uri + ' to ' + result.message)
        else:
            raise ValueError('Unknown status %s.' % result.status)

    def write_entry(self, what: str, docname: str, filename: str, line: int, uri: str) -> None:
        self.txt_outfile.write('%s:%s: [%s] %s\n' % (filename, line, what, uri))

    def write_linkstat(self, data: dict) -> None:
        self.json_outfile.write(json.dumps(data))
        self.json_outfile.write('\n')

    def finish(self) -> None:
        checker = HyperlinkAvailabilityChecker(self.env, self.config)
        logger.info('')
        output_text = path.join(self.outdir, 'output.txt')
        output_json = path.join(self.outdir, 'output.json')
        with open(output_text, 'w', encoding='utf-8') as self.txt_outfile, open(output_json, 'w', encoding='utf-8') as self.json_outfile:
            for result in checker.check(self.hyperlinks):
                self.process_result(result)
        if self.broken_hyperlinks:
            self.app.statuscode = 1