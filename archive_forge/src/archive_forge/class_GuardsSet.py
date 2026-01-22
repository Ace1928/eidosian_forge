from __future__ import annotations
import contextlib
import dataclasses
import enum
import functools
import logging
import threading
import traceback
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
import torch
from torch.utils import _pytree as pytree
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakTensorKeyDictionary
class GuardsSet:

    def __init__(self, inner=None):
        if inner is None:
            inner = set()
        self.inner = inner

    def __iter__(self):
        return iter(self.inner)

    def __len__(self):
        return len(self.inner)

    def __sub__(self, other):
        return GuardsSet(self.inner - other.inner)

    def __bool__(self):
        return bool(self.inner)

    def add(self, guard: Guard, *, skip=0):
        if guard in self.inner:
            return
        if guard.stack is None:
            guard.stack = CapturedTraceback.extract(skip=1 + skip)
        if guard.user_stack is None:
            guard.user_stack = TracingContext.extract_stack()
        self.inner.add(guard)

    def update(self, *others: Set[Guard]):
        for o in others:
            for g in o:
                self.add(g, skip=1)