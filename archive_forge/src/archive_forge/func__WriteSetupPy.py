import argparse
import contextlib
import io
import json
import logging
import os
import pkgutil
import sys
from apitools.base.py import exceptions
from apitools.gen import gen_client_lib
from apitools.gen import util
def _WriteSetupPy(codegen):
    with io.open('setup.py', 'w') as out:
        codegen.WriteSetupPy(out)