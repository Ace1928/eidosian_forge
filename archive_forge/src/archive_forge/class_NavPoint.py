import html
from os import path
from typing import Any, Dict, List, NamedTuple, Set, Tuple
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import _epub_base
from sphinx.config import ENUM, Config
from sphinx.locale import __
from sphinx.util import logging, xmlname_checker
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import make_filename
class NavPoint(NamedTuple):
    text: str
    refuri: str
    children: List[Any]