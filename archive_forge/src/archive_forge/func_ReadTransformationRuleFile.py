from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.backup_restore import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def ReadTransformationRuleFile(file_arg):
    """Reads content of the transformation rule file specified in file_arg."""
    if not file_arg:
        return None
    data = console_io.ReadFromFileOrStdin(file_arg, binary=False)
    ms = api_util.GetMessagesModule()
    temp_restore_config = export_util.Import(message_type=ms.RestoreConfig, stream=data, schema_path=export_util.GetSchemaPath('gkebackup', 'v1', 'TransformationRules'))
    return temp_restore_config.transformationRules