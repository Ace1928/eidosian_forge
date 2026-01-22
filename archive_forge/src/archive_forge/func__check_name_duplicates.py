from __future__ import annotations
import traceback
import warnings
from collections import defaultdict
from types import ModuleType
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Tuple, Type
from zope.interface import implementer
from scrapy import Request, Spider
from scrapy.interfaces import ISpiderLoader
from scrapy.settings import BaseSettings
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes
def _check_name_duplicates(self) -> None:
    dupes = []
    for name, locations in self._found.items():
        dupes.extend([f'  {cls} named {name!r} (in {mod})' for mod, cls in locations if len(locations) > 1])
    if dupes:
        dupes_string = '\n\n'.join(dupes)
        warnings.warn(f'There are several spiders with the same name:\n\n{dupes_string}\n\n  This can cause unexpected behavior.', category=UserWarning)