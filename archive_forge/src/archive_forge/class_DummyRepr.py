import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
class DummyRepr(object):

    def __repr__(self):
        return 'DummyRepr'

    def _repr_html_(self):
        return '<b>dummy</b>'

    def _repr_javascript_(self):
        return ("console.log('hi');", {'key': 'value'})