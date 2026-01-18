from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
def utctz_to_altz(utctz: str) -> int:
    """Convert a git timezone offset into a timezone offset west of
    UTC in seconds (compatible with :attr:`time.altzone`).

    :param utctz: git utc timezone string, e.g. +0200
    """
    int_utctz = int(utctz)
    seconds = abs(int_utctz) // 100 * 3600 + abs(int_utctz) % 100 * 60
    return seconds if int_utctz < 0 else -seconds