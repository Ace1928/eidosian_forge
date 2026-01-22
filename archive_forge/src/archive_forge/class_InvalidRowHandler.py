import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
class InvalidRowHandler:

    def __init__(self, result):
        self.result = result
        self.rows = []

    def __call__(self, row):
        self.rows.append(row)
        return self.result

    def __eq__(self, other):
        return isinstance(other, InvalidRowHandler) and other.result == self.result

    def __ne__(self, other):
        return not isinstance(other, InvalidRowHandler) or other.result != self.result