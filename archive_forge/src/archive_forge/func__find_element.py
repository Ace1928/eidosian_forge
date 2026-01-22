import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
def _find_element(self, tag, href_attr, href_extract, content, id, href_pattern, html_pattern, index, verbose):
    content_pat = _make_pattern(content)
    id_pat = _make_pattern(id)
    href_pat = _make_pattern(href_pattern)
    html_pat = _make_pattern(html_pattern)
    _tag_re = re.compile('<%s\\s+(.*?)>(.*?)</%s>' % (tag, tag), re.I + re.S)

    def printlog(s):
        if verbose:
            print(s)
    found_links = []
    total_links = 0
    for match in _tag_re.finditer(self.body):
        el_html = match.group(0)
        el_attr = match.group(1)
        el_content = match.group(2)
        attrs = _parse_attrs(el_attr)
        if verbose:
            printlog('Element: %r' % el_html)
        if not attrs.get(href_attr):
            printlog('  Skipped: no %s attribute' % href_attr)
            continue
        el_href = attrs[href_attr]
        if href_extract:
            m = href_extract.search(el_href)
            if not m:
                printlog("  Skipped: doesn't match extract pattern")
                continue
            el_href = m.group(1)
        attrs['uri'] = el_href
        if el_href.startswith('#'):
            printlog('  Skipped: only internal fragment href')
            continue
        if el_href.startswith('javascript:'):
            printlog('  Skipped: cannot follow javascript:')
            continue
        total_links += 1
        if content_pat and (not content_pat(el_content)):
            printlog("  Skipped: doesn't match description")
            continue
        if id_pat and (not id_pat(attrs.get('id', ''))):
            printlog("  Skipped: doesn't match id")
            continue
        if href_pat and (not href_pat(el_href)):
            printlog("  Skipped: doesn't match href")
            continue
        if html_pat and (not html_pat(el_html)):
            printlog("  Skipped: doesn't match html")
            continue
        printlog('  Accepted')
        found_links.append((el_html, el_content, attrs))
    if not found_links:
        raise IndexError('No matching elements found (from %s possible)' % total_links)
    if index is None:
        if len(found_links) > 1:
            raise IndexError('Multiple links match: %s' % ', '.join([repr(anc) for anc, d, attr in found_links]))
        found_link = found_links[0]
    else:
        try:
            found_link = found_links[index]
        except IndexError:
            raise IndexError('Only %s (out of %s) links match; index %s out of range' % (len(found_links), total_links, index))
    return found_link