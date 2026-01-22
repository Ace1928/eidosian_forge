import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
class CommandLineInputSpec2(nib.CommandLineInputSpec):
    foo = nib.File(argstr='%s', desc='a str', genfile=True)