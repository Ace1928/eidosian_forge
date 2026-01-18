import json
import os
import subprocess
import sys
import warnings
from argparse import ArgumentParser
from contextlib import AbstractContextManager
from typing import Dict, List, Optional
import requests
from ..utils import logging
from . import BaseTransformersCLICommand
def write_msg(msg: Dict):
    """Write out the message in Line delimited JSON."""
    msg = json.dumps(msg) + '\n'
    sys.stdout.write(msg)
    sys.stdout.flush()