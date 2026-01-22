import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class StyleConfig(object):
    """Configuration class from elyxer.config file"""
    hspaces = {'\\enskip{}': '\u2002', '\\hfill{}': '<span class="hfill"> </span>', '\\hspace*{\\fill}': '\u2003', '\\hspace*{}': '', '\\hspace{}': '\u2003', '\\negthinspace{}': '', '\\qquad{}': '\u2003\u2003', '\\quad{}': '\u2003', '\\space{}': '\xa0', '\\thinspace{}': '\u2009', '~': '\xa0'}
    quotes = {'ald': '»', 'als': '›', 'ard': '«', 'ars': '‹', 'eld': '&ldquo;', 'els': '&lsquo;', 'erd': '&rdquo;', 'ers': '&rsquo;', 'fld': '«', 'fls': '‹', 'frd': '»', 'frs': '›', 'gld': '„', 'gls': '‚', 'grd': '“', 'grs': '‘', 'pld': '„', 'pls': '‚', 'prd': '”', 'prs': '’', 'sld': '”', 'srd': '”'}
    referenceformats = {'eqref': '(@↕)', 'formatted': '¶↕', 'nameref': '$↕', 'pageref': '#↕', 'ref': '@↕', 'vpageref': 'on-page#↕', 'vref': '@on-page#↕'}
    size = {'ignoredtexts': ['col', 'text', 'line', 'page', 'theight', 'pheight']}
    vspaces = {'bigskip': '<div class="bigskip"> </div>', 'defskip': '<div class="defskip"> </div>', 'medskip': '<div class="medskip"> </div>', 'smallskip': '<div class="smallskip"> </div>', 'vfill': '<div class="vfill"> </div>'}