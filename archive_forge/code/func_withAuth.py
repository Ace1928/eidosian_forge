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
def withAuth(self, auth: Optional['Auth']) -> 'Requester':
    """
        Create a new requester instance with identical configuration but the given authentication method.
        :param auth: authentication method
        :return: new Requester implementation
        """
    kwargs = self.kwargs
    kwargs.update(auth=auth)
    return Requester(**kwargs)