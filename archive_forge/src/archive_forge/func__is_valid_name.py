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
def _is_valid_name(self, project_name):

    def _module_exists(module_name):
        spec = find_spec(module_name)
        return spec is not None and spec.loader is not None
    if not re.search('^[_a-zA-Z]\\w*$', project_name):
        print('Error: Project names must begin with a letter and contain only\nletters, numbers and underscores')
    elif _module_exists(project_name):
        print(f'Error: Module {project_name!r} already exists')
    else:
        return True
    return False