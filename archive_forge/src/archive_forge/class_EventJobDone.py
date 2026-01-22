import asyncio
import concurrent.futures
import logging
import queue
import sys
import threading
from typing import (
from wandb.errors.term import termerror
from wandb.filesync import upload_job
from wandb.sdk.lib.paths import LogicalPath
class EventJobDone(NamedTuple):
    job: RequestUpload
    exc: Optional[BaseException]