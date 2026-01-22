from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags
from googlecloudsdk.command_lib.compute.security_policies import security_policies_utils
from googlecloudsdk.core.util import files
import six
def _GetTemplateFromFile(self, args, messages):
    if not os.path.exists(args.file_name):
        raise exceptions.BadFileException('No such file [{0}]'.format(args.file_name))
    if os.path.isdir(args.file_name):
        raise exceptions.BadFileException('[{0}] is a directory'.format(args.file_name))
    try:
        with files.FileReader(args.file_name) as import_file:
            if args.file_format == 'json':
                return security_policies_utils.SecurityPolicyFromFile(import_file, messages, 'json')
            return security_policies_utils.SecurityPolicyFromFile(import_file, messages, 'yaml')
    except Exception as exp:
        exp_msg = getattr(exp, 'message', six.text_type(exp))
        msg = 'Unable to read security policy config from specified file [{0}] because [{1}]'.format(args.file_name, exp_msg)
        raise exceptions.BadFileException(msg)