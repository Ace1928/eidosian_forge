import datetime
import functools
import hashlib
import json
import logging
import optparse
import os.path
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.rich.console import Group
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import get_default_environment
from pip._internal.metadata.base import DistributionVersion
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.entrypoints import (
from pip._internal.utils.filesystem import adjacent_tmp_file, check_path_owner, replace
from pip._internal.utils.misc import ensure_dir
Check if we have a not-outdated version loaded already.