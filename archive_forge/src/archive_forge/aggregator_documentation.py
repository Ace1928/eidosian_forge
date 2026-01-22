from __future__ import annotations
import argparse
import configparser
import logging
from typing import Sequence
from flake8.options import config
from flake8.options.manager import OptionManager
Aggregate and merge CLI and config file options.