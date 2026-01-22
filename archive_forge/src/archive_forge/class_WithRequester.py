import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException
class WithRequester(Generic[T]):
    """
    Mixin class that allows to set a requester.
    """
    __requester: Requester

    def __init__(self) -> None:
        self.__requester: Optional[Requester] = None

    @property
    def requester(self) -> Requester:
        return self.__requester

    def withRequester(self, requester: Requester) -> 'WithRequester[T]':
        assert isinstance(requester, Requester), requester
        self.__requester = requester
        return self