import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition

        Assert that the code inside the with block is not deprecated.  Intended
        for testing uses of traits.util.deprecated.deprecated.

        