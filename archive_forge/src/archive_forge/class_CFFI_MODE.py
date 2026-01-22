import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
class CFFI_MODE(enum.Enum):
    API = 'API'
    ABI = 'ABI'
    BOTH = 'BOTH'
    ANY = 'ANY'