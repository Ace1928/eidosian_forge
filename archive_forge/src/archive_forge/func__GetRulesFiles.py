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
def _GetRulesFiles(self, config_files):
    """Returns the rules files to import rules from."""
    rules_files = []
    for config_file in config_files:
        try:
            data = files.ReadFileContents(config_file)
        except files.MissingFileError:
            raise exceptions.BadArgumentException('--config-flies', 'specified file [{}] does not exist.'.format(config_file))
        rules_files.append(self.messages.RulesFile(rulesContent=data, rulesSourceFilename=os.path.basename(config_file)))
    return rules_files