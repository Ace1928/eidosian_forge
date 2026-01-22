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
def _spider_exists(self, name: str) -> bool:
    if not self.settings.get('NEWSPIDER_MODULE'):
        path = Path(name + '.py')
        if path.exists():
            print(f'{path.resolve()} already exists')
            return True
        return False
    assert self.crawler_process is not None, 'crawler_process must be set before calling run'
    try:
        spidercls = self.crawler_process.spider_loader.load(name)
    except KeyError:
        pass
    else:
        print(f'Spider {name!r} already exists in module:')
        print(f'  {spidercls.__module__}')
        return True
    spiders_module = import_module(self.settings['NEWSPIDER_MODULE'])
    spiders_dir = Path(cast(str, spiders_module.__file__)).parent
    spiders_dir_abs = spiders_dir.resolve()
    path = spiders_dir_abs / (name + '.py')
    if path.exists():
        print(f'{path} already exists')
        return True
    return False