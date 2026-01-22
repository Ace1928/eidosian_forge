import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
class CommandLineInputSpec1(nib.CommandLineInputSpec):
    foo = nib.Str(argstr='%s', desc='a str')
    goo = nib.traits.Bool(argstr='-g', desc='a bool', position=0)
    hoo = nib.traits.List(argstr='-l %s', desc='a list')
    moo = nib.traits.List(argstr='-i %d...', desc='a repeated list', position=-1)
    noo = nib.traits.Int(argstr='-x %d', desc='an int')
    roo = nib.traits.Str(desc='not on command line')
    soo = nib.traits.Bool(argstr='-soo')