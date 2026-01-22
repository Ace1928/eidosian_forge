import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import urlopen
import pytest
import modin.pandas
from modin.utils import PANDAS_API_URL_TEMPLATE
def _test_url(url):
    try:
        with urlopen(url):
            pass
    except HTTPError:
        broken.append(url)