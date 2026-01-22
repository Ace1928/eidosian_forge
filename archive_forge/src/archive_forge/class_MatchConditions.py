from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class MatchConditions(enum.Flag):
    """Acceptable condition(s) for matching user input to available choices."""
    CHOICE = enum.auto()
    'Match any choice.'
    ANY = enum.auto()
    'Match any non-empty string.'
    NOTHING = enum.auto()
    'Match an empty string which is not followed by a boundary match.'