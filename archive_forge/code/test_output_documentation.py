from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
Actually run trial as if on the command line and check that the output
    is what we expect.
    