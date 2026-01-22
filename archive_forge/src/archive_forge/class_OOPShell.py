import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
class OOPShell(nib.CommandLine):
    _cmd_prefix = 'bash {}/'.format(oop)