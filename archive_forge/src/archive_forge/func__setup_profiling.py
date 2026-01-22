from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
@post
def _setup_profiling(options, file_config):
    from sqlalchemy.testing import profiling
    profiling._profile_stats = profiling.ProfileStatsFile(file_config.get('sqla_testing', 'profile_file'), sort=options.profilesort, dump=options.profiledump)