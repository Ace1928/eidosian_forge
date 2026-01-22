import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
class VersionDict(TypedDict):
    major: str
    minor: str
    build_number: str