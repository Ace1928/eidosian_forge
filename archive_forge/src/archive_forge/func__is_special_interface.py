import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def _is_special_interface(self, interface: type) -> bool:
    return any((_is_specialization(interface, cls) for cls in [AssistedBuilder, ProviderOf]))