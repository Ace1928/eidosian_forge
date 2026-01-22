from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
class _BaseImport(object):
    """An imported DM config object."""

    def __init__(self, full_path, name):
        self.full_path = full_path
        self.name = name
        self.content = None
        self.base_name = None

    def GetFullPath(self):
        return self.full_path

    def GetName(self):
        return self.name

    def SetContent(self, content):
        self.content = content
        return self

    def IsTemplate(self):
        return self.full_path.endswith(('.jinja', '.py'))