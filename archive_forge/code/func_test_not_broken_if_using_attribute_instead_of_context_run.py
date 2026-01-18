from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
def test_not_broken_if_using_attribute_instead_of_context_run(self):
    let1 = greenlet(getcurrent().switch)
    let2 = greenlet(getcurrent().switch)
    let1.gr_context = copy_context()
    let2.gr_context = copy_context()
    let1.switch()
    let2.switch()
    let1.switch()
    let2.switch()