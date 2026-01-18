import os
import shutil
import string
from importlib import import_module
from pathlib import Path
from typing import Optional, cast
from urllib.parse import urlparse
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase
Generate the spider module, based on the given template