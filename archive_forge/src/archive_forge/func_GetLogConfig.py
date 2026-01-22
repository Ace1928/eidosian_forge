from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.service_extensions import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
def GetLogConfig(parsed_dict):
    log_config_dict = {_ConvertToCamelCase(key): parsed_dict[key] for key, value in parsed_dict.items()}
    return encoding.DictToMessage(log_config_dict, messages.WasmPluginLogConfig)