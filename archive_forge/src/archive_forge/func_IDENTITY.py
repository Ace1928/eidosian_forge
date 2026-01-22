import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def IDENTITY(monitor):
    return monitor