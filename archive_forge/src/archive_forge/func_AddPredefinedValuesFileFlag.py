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
def AddPredefinedValuesFileFlag(parser):
    """Adds a flag for the predefined x509 extensions file for a Certificate Template."""
    base.Argument('--predefined-values-file', action='store', type=arg_parsers.YAMLFileContents(), help='A YAML file describing any predefined X.509 values set by this template. The provided extensions will be copied over to any certificate requests that use this template, taking precedent over any allowed extensions in the certificate request. The format of this file should be a YAML representation of the X509Parameters message, which is defined here: https://cloud.google.com/certificate-authority-service/docs/reference/rest/v1/X509Parameters. Some examples can be found here: https://cloud.google.com/certificate-authority-service/docs/creating-certificate-template').AddToParser(parser)