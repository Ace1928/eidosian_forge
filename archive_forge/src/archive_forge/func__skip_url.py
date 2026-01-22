import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
import glob
import pytest
def _skip_url(url: str):
    if url in SKIP_URLS:
        return True
    if url.startswith('https://github.com/holoviz/hvplot/pull/'):
        return True
    if url.startswith('https://img.shields.io'):
        return True
    if url.startswith('assets.holoviews.org/data/'):
        return True
    if url.startswith('Math.PI'):
        return True
    return False