import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def ConversionHeader(i: str, filename: OptStr='unknown'):
    t = i.lower()
    import textwrap
    html = textwrap.dedent('\n            <!DOCTYPE html>\n            <html>\n            <head>\n            <style>\n            body{background-color:gray}\n            div{position:relative;background-color:white;margin:1em auto}\n            p{position:absolute;margin:0}\n            img{position:absolute}\n            </style>\n            </head>\n            <body>\n            ')
    xml = textwrap.dedent('\n            <?xml version="1.0"?>\n            <document name="%s">\n            ' % filename)
    xhtml = textwrap.dedent('\n            <?xml version="1.0"?>\n            <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n            <html xmlns="http://www.w3.org/1999/xhtml">\n            <head>\n            <style>\n            body{background-color:gray}\n            div{background-color:white;margin:1em;padding:1em}\n            p{white-space:pre-wrap}\n            </style>\n            </head>\n            <body>\n            ')
    text = ''
    json = '{"document": "%s", "pages": [\n' % filename
    if t == 'html':
        r = html
    elif t == 'json':
        r = json
    elif t == 'xml':
        r = xml
    elif t == 'xhtml':
        r = xhtml
    else:
        r = text
    return r