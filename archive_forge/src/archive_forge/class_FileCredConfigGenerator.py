from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class FileCredConfigGenerator(CredConfigGenerator):
    """The generator for File-based credential configs."""

    def __init__(self, config_type, credential_source_file):
        super(FileCredConfigGenerator, self).__init__(config_type)
        self.credential_source_file = credential_source_file

    def get_source(self, args):
        credential_source = {'file': self.credential_source_file}
        token_format = self._get_format(args.credential_source_type, args.credential_source_field_name)
        if token_format:
            credential_source['format'] = token_format
        return credential_source