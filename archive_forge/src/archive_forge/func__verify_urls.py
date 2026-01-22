import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
import glob
import pytest
def _verify_urls(urls):
    """Returns True if all urls are valid"""
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for url in urls:
            futures[executor.submit(_request_a_response, url)] = url
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
            except Exception as ex:
                raise ValueError(f'The url {url} raised an exception') from ex
            if not result.status == 200:
                raise ValueError(f'The url {url} responded with status {result.status}, not 200.')
        return True