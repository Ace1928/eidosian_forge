import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
import glob
import pytest
def _request_a_response(url):
    return urlopen(url)