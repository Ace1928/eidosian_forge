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
Facade for pytest.mark.parametrize.

        Automatically derives argument names from the callable which in our
        case is always a method on a class with positional arguments.

        ids for parameter sets are derived using an optional template.

        