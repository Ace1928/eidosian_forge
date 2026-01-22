from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def ValidateSubjectConfig(subject_config, is_ca):
    """Validates a SubjectConfig object."""
    san_names = []
    if subject_config.subjectAltName:
        san_names = [subject_config.subjectAltName.emailAddresses, subject_config.subjectAltName.dnsNames, subject_config.subjectAltName.ipAddresses, subject_config.subjectAltName.uris]
    if not subject_config.subject.commonName and all([not elem for elem in san_names]):
        raise exceptions.InvalidArgumentException('--subject', 'The certificate you are creating does not contain a common name or a subject alternative name.')
    if is_ca and (not subject_config.subject.organization):
        raise exceptions.InvalidArgumentException('--subject', 'An organization must be provided for a certificate authority certificate.')