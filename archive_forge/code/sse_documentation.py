import io
import logging
import re
from datetime import datetime, timezone
from functools import partial
from typing import (
import anyio
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
Setter for ping_interval property.

        :param int value: interval in sec between two ping values.
        