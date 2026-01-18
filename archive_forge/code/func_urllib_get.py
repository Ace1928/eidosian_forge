from __future__ import print_function
import sys
import time
import urllib
import urllib3  # noqa: E402
def urllib_get(url_list):
    assert url_list
    for url in url_list:
        now = time.time()
        urllib.urlopen(url)
        elapsed = time.time() - now
        print('Got in %0.3f: %s' % (elapsed, url))