import os
import webbrowser
from urllib.parse import quote, urlunparse
def _iterpath(path):
    path, last = os.path.split(path)
    if last:
        yield from _iterpath(path)
        yield last