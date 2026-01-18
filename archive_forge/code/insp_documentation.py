from __future__ import absolute_import
import code
import logging
import sys
import collections
import warnings
import numpy as np
import click
from . import options
import rasterio
from rasterio.plot import show, show_hist
Open the input file in a Python interpreter.