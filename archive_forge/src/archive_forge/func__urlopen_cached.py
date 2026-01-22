from statsmodels.compat.python import lrange
from io import StringIO
from os import environ, makedirs
from os.path import abspath, dirname, exists, expanduser, join
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen
import numpy as np
from pandas import Index, read_csv, read_stata
def _urlopen_cached(url, cache):
    """
    Tries to load data from cache location otherwise downloads it. If it
    downloads the data and cache is not None then it will put the downloaded
    data in the cache path.
    """
    from_cache = False
    if cache is not None:
        file_name = url.split('://')[-1].replace('/', ',')
        file_name = file_name.split('.')
        if len(file_name) > 1:
            file_name[-2] += '-v2'
        else:
            file_name[0] += '-v2'
        file_name = '.'.join(file_name) + '.zip'
        cache_path = join(cache, file_name)
        try:
            data = _open_cache(cache_path)
            from_cache = True
        except:
            pass
    if not from_cache:
        data = urlopen(url, timeout=3).read()
        if cache is not None:
            _cache_it(data, cache_path)
    return (data, from_cache)