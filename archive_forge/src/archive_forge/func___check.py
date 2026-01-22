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
def __check(self, status: int, responseHeaders: Dict[str, Any], output: str) -> Tuple[Dict[str, Any], Any]:
    data = self.__structuredFromJson(output)
    if status >= 400:
        raise self.createException(status, responseHeaders, data)
    return (responseHeaders, data)