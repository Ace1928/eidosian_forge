import base64
import copy
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from rdflib.query import Result
from rdflib.term import BNode

        auth, if present, must be a tuple of (username, password) used for Basic Authentication

        Any additional keyword arguments will be passed to to the request, and can be used to setup timesouts etc.
        