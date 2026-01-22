from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dns import dns_keys
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
import six
def _GenerateDSRecord(key):
    key_tag = six.text_type(key.keyTag)
    key_algorithm = six.text_type(ALGORITHM_NUMBERS[key.algorithm.name])
    digest_algorithm = six.text_type(DIGEST_TYPE_NUMBERS[key.digests[0].type.name])
    digest = key.digests[0].digest
    return ' '.join([key_tag, key_algorithm, digest_algorithm, digest])