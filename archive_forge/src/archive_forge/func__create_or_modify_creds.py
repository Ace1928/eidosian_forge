from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_creds(transfer_spec, args, messages):
    """Creates or modifies TransferSpec source creds based on args."""
    if transfer_spec.awsS3DataSource:
        if getattr(args, 'source_creds_file', None):
            access_key_id, secret_access_key, role_arn = creds_util.get_aws_creds_from_file(args.source_creds_file)
        else:
            log.warning('No --source-creds-file flag. Checking system config files for AWS credentials.')
            access_key_id, secret_access_key = creds_util.get_default_aws_creds()
            role_arn = None
        if not (access_key_id and secret_access_key or role_arn):
            log.warning('Missing AWS source creds.')
        transfer_spec.awsS3DataSource.awsAccessKey = messages.AwsAccessKey(accessKeyId=access_key_id, secretAccessKey=secret_access_key)
        transfer_spec.awsS3DataSource.roleArn = role_arn
    elif transfer_spec.azureBlobStorageDataSource:
        if getattr(args, 'source_creds_file', None):
            sas_token = creds_util.get_values_for_keys_from_file(args.source_creds_file, ['sasToken'])['sasToken']
        else:
            log.warning('No Azure source creds set. Consider adding --source-creds-file flag.')
            sas_token = None
        transfer_spec.azureBlobStorageDataSource.azureCredentials = messages.AzureCredentials(sasToken=sas_token)