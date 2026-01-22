from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
def ChangeFromYamlFile(yaml_file, api_version='v1'):
    """Returns the change contained in the given yaml file.

  Args:
    yaml_file: file, A yaml file with change.
    api_version: [str], the api version to use for creating the change object.

  Returns:
    Change, the change contained in the given yaml file.

  Raises:
    CorruptedTransactionFileError: if the record_set_dictionaries are invalid
  """
    messages = apis.GetMessagesModule('dns', api_version)
    try:
        change_dict = yaml.load(yaml_file) or {}
    except yaml.YAMLParseError:
        raise CorruptedTransactionFileError()
    if change_dict.get('additions') is None or change_dict.get('deletions') is None:
        raise CorruptedTransactionFileError()
    change = messages.Change()
    change.additions = _RecordSetsFromDictionaries(messages, change_dict['additions'])
    change.deletions = _RecordSetsFromDictionaries(messages, change_dict['deletions'])
    return change