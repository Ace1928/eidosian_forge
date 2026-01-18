from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
Skip test completely if no Docker-based XNAT instance available
    