from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import functools
import json
import re
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.api_lib.functions.v1 import operations
from googlecloudsdk.api_lib.functions.v2 import util as v2_util
from googlecloudsdk.api_lib.storage import storage_api as gcs_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as base_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.generated_clients.apis.cloudfunctions.v1 import cloudfunctions_v1_messages
import six.moves.http_client
def ValidateRuntimeOrRaise(client, runtime, region):
    """Checks if runtime is supported.

  Does not raise if the runtime list cannot be retrieved

  Args:
    client: v2 GCF client that supports ListRuntimes()
    runtime: str, the runtime.
    region: str, region code.

  Returns:
    warning: None|str, the warning if deprecated
  """
    response = client.ListRuntimes(region, query_filter='name={} AND environment={}'.format(runtime, client.messages.Runtime.EnvironmentValueValuesEnum.GEN_1))
    if not response or response.runtimes is None:
        return None
    if len(response.runtimes) < 1:
        raise exceptions.FunctionsError('argument `--runtime`: {} is not a supported runtime on GCF 1st gen. Use `gcloud functions runtimes list` to get a list of available runtimes'.format(runtime))
    runtime_info = response.runtimes[0]
    return runtime_info.warnings[0] if runtime_info and runtime_info.warnings else None