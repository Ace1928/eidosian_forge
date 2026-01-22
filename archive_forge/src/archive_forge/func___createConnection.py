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
def __createConnection(self) -> Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]:
    if self.__persist and self.__connection is not None:
        return self.__connection
    with self.__connection_lock:
        if self.__connection is not None:
            if self.__persist:
                return self.__connection
            self.__connection.close()
        self.__connection = self.__connectionClass(self.__hostname, self.__port, retry=self.__retry, pool_size=self.__pool_size, timeout=self.__timeout, verify=self.__verify)
    return self.__connection