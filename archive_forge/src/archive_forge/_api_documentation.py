from __future__ import annotations
import contextlib
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import local
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakValueDictionary
from ._error import Timeout
Called when the lock object is deleted.