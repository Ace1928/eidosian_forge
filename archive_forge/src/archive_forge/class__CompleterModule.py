from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import walker
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import exceptions as cache_exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.cache import resource_cache
import six
class _CompleterModule(object):

    def __init__(self, module_path, collection, api_version, completer_type):
        self.module_path = module_path
        self.collection = collection
        self.api_version = api_version
        self.type = completer_type
        self.attachments = []
        self._attachments_dict = {}