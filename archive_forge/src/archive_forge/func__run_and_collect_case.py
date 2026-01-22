import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
def _run_and_collect_case(case, res):
    """Run test case against result and use weakref to drop the refcount"""
    case.run(res)
    return weakref.ref(case)