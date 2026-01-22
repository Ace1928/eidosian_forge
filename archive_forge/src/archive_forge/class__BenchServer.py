import subprocess
import sys
import time
from urllib.parse import urlencode
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.linkextractors import LinkExtractor
class _BenchServer:

    def __enter__(self):
        from scrapy.utils.test import get_testenv
        pargs = [sys.executable, '-u', '-m', 'scrapy.utils.benchserver']
        self.proc = subprocess.Popen(pargs, stdout=subprocess.PIPE, env=get_testenv())
        self.proc.stdout.readline()

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.kill()
        self.proc.wait()
        time.sleep(0.2)