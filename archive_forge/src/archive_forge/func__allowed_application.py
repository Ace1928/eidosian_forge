from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _allowed_application(parser):
    base.Argument('--allowed-application', type=arg_parsers.ArgDict(spec={'sha1_fingerprint': str, 'package_name': str}, required_keys=['sha1_fingerprint', 'package_name'], max_length=2), metavar='sha1_fingerprint=SHA1_FINGERPRINT,package_name=PACKAGE_NAME', action='append', help='Repeatable. Specify multiple allowed applications. The accepted keys are `sha1_fingerprint` and `package_name`.').AddToParser(parser)