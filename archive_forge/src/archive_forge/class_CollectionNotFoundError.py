from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
class CollectionNotFoundError(core_exceptions.Error):
    """Exception for attempts to generate unsupported commands."""

    def __init__(self, collection):
        message = '{collection} collection is not found'.format(collection=collection)
        super(CollectionNotFoundError, self).__init__(message)