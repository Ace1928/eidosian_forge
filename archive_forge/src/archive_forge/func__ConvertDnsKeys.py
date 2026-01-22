from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.dns import util as dns_api_util
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import six
def _ConvertDnsKeys(domains_messages, dns_messages, dns_keys):
    """Converts DnsKeys to DsRecords."""
    ds_records = []
    for key in dns_keys:
        if key.type != dns_messages.DnsKey.TypeValueValuesEnum.keySigning:
            continue
        if not key.isActive:
            continue
        try:
            algorithm = domains_messages.DsRecord.AlgorithmValueValuesEnum(six.text_type(key.algorithm).upper())
            for d in key.digests:
                digest_type = domains_messages.DsRecord.DigestTypeValueValuesEnum(six.text_type(d.type).upper())
                ds_records.append(domains_messages.DsRecord(keyTag=key.keyTag, digest=d.digest, algorithm=algorithm, digestType=digest_type))
        except TypeError:
            continue
    return ds_records