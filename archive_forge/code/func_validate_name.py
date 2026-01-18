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
def validate_name(self, name, name_or_dir='name'):
    if name is None:
        raise CommandError('you must provide {an} {app} name'.format(an=self.a_or_an, app=self.app_or_project))
    if not name.isidentifier():
        raise CommandError("'{name}' is not a valid {app} {type}. Please make sure the {type} is a valid identifier.".format(name=name, app=self.app_or_project, type=name_or_dir))
    if find_spec(name) is not None:
        raise CommandError("'{name}' conflicts with the name of an existing Python module and cannot be used as {an} {app} {type}. Please try another {type}.".format(name=name, an=self.a_or_an, app=self.app_or_project, type=name_or_dir))