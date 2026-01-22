import argparse
import mimetypes
import os
import posixpath
import shutil
import stat
import tempfile
from importlib.util import find_spec
from urllib.request import build_opener
import django
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import (
from django.template import Context, Engine
from django.utils import archive
from django.utils.http import parse_header_parameters
from django.utils.version import get_docs_version
def apply_umask(self, old_path, new_path):
    current_umask = os.umask(0)
    os.umask(current_umask)
    current_mode = stat.S_IMODE(os.stat(old_path).st_mode)
    os.chmod(new_path, current_mode & ~current_umask)