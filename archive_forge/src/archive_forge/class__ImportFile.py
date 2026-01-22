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
class _ImportFile(_BaseImport):
    """Performs common operations on an imported file."""

    def __init__(self, full_path, name=None):
        name = name if name else full_path
        super(_ImportFile, self).__init__(full_path, name)

    def GetBaseName(self):
        if self.base_name is None:
            self.base_name = os.path.basename(self.full_path)
        return self.base_name

    def Exists(self):
        return os.path.isfile(self.full_path)

    def GetContent(self):
        if self.content is None:
            try:
                self.content = files.ReadFileContents(self.full_path)
            except files.Error as e:
                raise exceptions.ConfigError("Unable to read file '%s'. %s" % (self.full_path, six.text_type(e)))
        return self.content

    def BuildChildPath(self, child_path):
        if _IsUrl(child_path):
            return child_path
        return os.path.normpath(os.path.join(os.path.dirname(self.full_path), child_path))