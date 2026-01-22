import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
class PickledBreaker:

    def __setstate__(self, d):
        raise Exception()