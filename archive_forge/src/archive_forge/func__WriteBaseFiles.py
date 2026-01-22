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
def _WriteBaseFiles(codegen):
    with util.Chdir(codegen.outdir):
        _CopyLocalFile('base_api.py')
        _CopyLocalFile('credentials_lib.py')
        _CopyLocalFile('exceptions.py')