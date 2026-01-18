from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from dns import rdatatype
from googlecloudsdk.api_lib.dns import import_util
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
Creates and returns a record-set from the given args.

  Args:
    args: The arguments to use to create the record-set.
    project: The GCP project where these resources are to be created.
    api_version: [str], the api version to use for creating the RecordSet.
    allow_extended_records: [bool], enables extended records if true, otherwise
      throws an exception when given an extended record type.

  Raises:
    UnsupportedRecordType: If given record-set type is not supported
    ForwardingRuleWithoutHealthCheck: If forwarding rules are specified without
      enabling health check.
    ForwardingRuleNotFound: Either the forwarding rule doesn't exist, or
      multiple forwarding rules present with the same name - across different
      regions.
    HealthCheckWithoutForwardingRule: Health check enabled but no forwarding
      rules present.

  Returns:
    ResourceRecordSet, the record-set created from the given args.
  