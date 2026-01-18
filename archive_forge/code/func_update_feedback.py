from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def update_feedback(self, feedback_id: ID_TYPE, *, score: Union[float, int, bool, None]=None, value: Union[float, int, bool, str, dict, None]=None, correction: Union[dict, None]=None, comment: Union[str, None]=None) -> None:
    """Update a feedback in the LangSmith API.

        Parameters
        ----------
        feedback_id : str or UUID
            The ID of the feedback to update.
        score : float or int or bool or None, default=None
            The score to update the feedback with.
        value : float or int or bool or str or dict or None, default=None
            The value to update the feedback with.
        correction : dict or None, default=None
            The correction to update the feedback with.
        comment : str or None, default=None
            The comment to update the feedback with.
        """
    feedback_update: Dict[str, Any] = {}
    if score is not None:
        feedback_update['score'] = score
    if value is not None:
        feedback_update['value'] = value
    if correction is not None:
        feedback_update['correction'] = correction
    if comment is not None:
        feedback_update['comment'] = comment
    response = self.session.patch(self.api_url + f'/feedback/{_as_uuid(feedback_id, 'feedback_id')}', headers={**self._headers, 'Content-Type': 'application/json'}, data=_dumps_json(feedback_update))
    ls_utils.raise_for_status_with_text(response)