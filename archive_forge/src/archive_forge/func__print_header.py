import argparse
import cProfile
import inspect
import os
import sys
from importlib.metadata import entry_points
import scrapy
from scrapy.commands import BaseRunSpiderCommand, ScrapyCommand, ScrapyHelpFormatter
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.misc import walk_modules
from scrapy.utils.project import get_project_settings, inside_project
from scrapy.utils.python import garbage_collect
def _print_header(settings, inproject):
    version = scrapy.__version__
    if inproject:
        print(f'Scrapy {version} - active project: {settings['BOT_NAME']}\n')
    else:
        print(f'Scrapy {version} - no active project\n')