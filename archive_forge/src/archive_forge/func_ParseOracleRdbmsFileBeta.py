from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import encoding as api_encoding
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.datastream import camel_case_utils
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
import six
def ParseOracleRdbmsFileBeta(messages, oracle_rdbms_file, release_track=base.ReleaseTrack.BETA):
    """Parses a oracle_rdbms_file into the OracleRdbms message. deprecated."""
    data = console_io.ReadFromFileOrStdin(oracle_rdbms_file, binary=False)
    try:
        oracle_rdbms_head_data = yaml.load(data)
    except Exception as e:
        raise ds_exceptions.ParseError('Cannot parse YAML:[{0}]'.format(e))
    oracle_rdbms_data = oracle_rdbms_head_data.get('oracle_rdbms', oracle_rdbms_head_data)
    return ParseOracleSchemasListToOracleRdbmsMessage(messages, oracle_rdbms_data, release_track)