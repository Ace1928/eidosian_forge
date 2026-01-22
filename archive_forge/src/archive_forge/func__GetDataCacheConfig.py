from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetDataCacheConfig(self, args):
    if args.IsKnownAndSpecified('enable_data_cache'):
        data_cache_config_obj = self.messages.DataCacheConfig
        return data_cache_config_obj(dataCacheEnabled=args.enable_data_cache)
    else:
        return None