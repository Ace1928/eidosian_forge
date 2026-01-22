import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
import glob
import pytest
def _find_urls(text):
    url = re.findall(URL_REGEX, text)
    return {_clean_url(x[0]) for x in url if not _skip_url(x[0])}