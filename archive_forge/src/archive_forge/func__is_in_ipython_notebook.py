import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def _is_in_ipython_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False