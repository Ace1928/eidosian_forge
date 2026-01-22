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
class UnsatisfiedRequirement(Error):
    """Requirement could not be satisfied."""

    def __init__(self, owner: Optional[object], interface: type) -> None:
        super().__init__(owner, interface)
        self.owner = owner
        self.interface = interface

    def __str__(self) -> str:
        on = '%s has an ' % _describe(self.owner) if self.owner else ''
        return '%sunsatisfied requirement on %s' % (on, _describe(self.interface))