from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def _GetRulesFormat(self, file_format):
    """Returns the file format enum to import rules from."""
    if file_format == 'ORA2PG':
        return self.messages.ImportMappingRulesRequest.RulesFormatValueValuesEnum.IMPORT_RULES_FILE_FORMAT_ORATOPG_CONFIG_FILE
    if file_format == 'HARBOUR_BRIDGE':
        return self.messages.ImportMappingRulesRequest.RulesFormatValueValuesEnum.IMPORT_RULES_FILE_FORMAT_HARBOUR_BRIDGE_SESSION_FILE
    return self.messages.ImportMappingRulesRequest.RulesFormatValueValuesEnum.IMPORT_RULES_FILE_FORMAT_UNSPECIFIED