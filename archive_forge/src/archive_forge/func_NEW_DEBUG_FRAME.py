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
def NEW_DEBUG_FRAME(self, requestHeader: Dict[str, str]) -> None:
    """
        Initialize a debug frame with requestHeader
        Frame count is updated and will be attached to respond header
        The structure of a frame: [requestHeader, statusCode, responseHeader, raw_data]
        Some of them may be None
        """
    if self.DEBUG_FLAG:
        new_frame = [requestHeader, None, None, None]
        if self._frameCount < self.DEBUG_FRAME_BUFFER_SIZE - 1:
            self._frameBuffer.append(new_frame)
        else:
            self._frameBuffer[0] = new_frame
        self._frameCount = len(self._frameBuffer) - 1