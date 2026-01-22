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
def _init_artifact(self, artifact_id: str) -> None:
    self._artifacts[artifact_id] = {'finalize': False, 'pending_count': 0, 'commit_requested': False, 'pre_commit_callbacks': set(), 'result_futures': set()}