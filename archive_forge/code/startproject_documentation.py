import os
import re
import string
from importlib.util import find_spec
from pathlib import Path
from shutil import copy2, copystat, ignore_patterns, move
from stat import S_IWUSR as OWNER_WRITE_PERMISSION
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase

        Since the original function always creates the directory, to resolve
        the issue a new function had to be created. It's a simple copy and
        was reduced for this case.

        More info at:
        https://github.com/scrapy/scrapy/pull/2005
        