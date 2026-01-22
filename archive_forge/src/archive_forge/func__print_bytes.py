import sys
from argparse import Namespace
from typing import List, Type
from w3lib.url import is_url
from scrapy import Spider
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.http import Request
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.spider import DefaultSpider, spidercls_for_request
def _print_bytes(self, bytes_):
    sys.stdout.buffer.write(bytes_ + b'\n')