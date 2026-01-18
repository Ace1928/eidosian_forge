from __future__ import annotations
import collections
import itertools
import json
import logging
import os
import platform
import sys
import warnings
from typing import TYPE_CHECKING
import requests
from monty.json import MontyDecoder
from pymatgen.core import SETTINGS
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        Args:
           *args: Pass through to either legacy or new MPRester.
           **kwargs: Pass through to either legacy or new MPRester.
        