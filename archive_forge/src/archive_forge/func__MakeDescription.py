from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def _MakeDescription(self):
    """Makes description for error by checking which fields are filled in."""
    if self.status_code and self.resource_item and self.instance_name:
        if self.status_code == 403:
            return 'User [{0}] does not have permission to access {1} [{2}] (or it may not exist)'.format(properties.VALUES.core.account.Get(), self.resource_item, self.instance_name)
        if self.status_code == 404:
            return '{0} [{1}] not found'.format(self.resource_item.capitalize(), self.instance_name)
        if self.status_code == 409:
            if self.resource_name == 'projects':
                return 'Resource in projects [{0}] is the subject of a conflict'.format(self.instance_name)
            else:
                return '{0} [{1}] is the subject of a conflict'.format(self.resource_item.capitalize(), self.instance_name)
    return super(HttpErrorPayload, self)._MakeDescription()