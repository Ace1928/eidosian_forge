import os
import re
from time import sleep
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger

        This is more or less the _submit_batchtask from sge.py with flipped
        variable names, different command line switches, and different output
        formatting/processing
        