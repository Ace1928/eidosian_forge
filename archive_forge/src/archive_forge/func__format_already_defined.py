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
def _format_already_defined(self, credential_source_type):
    if credential_source_type:
        raise GeneratorError('--credential-source-type is not supported with --azure or --aws')