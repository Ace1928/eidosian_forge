from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def _filter_exclusions(args):
    result = []
    gathered_exclusions = []
    for a in args:
        if isinstance(a, exclusions.compound):
            gathered_exclusions.append(a)
        else:
            result.append(a)
    return (result, gathered_exclusions)