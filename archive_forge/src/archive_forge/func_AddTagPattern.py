from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddTagPattern(parser):
    parser.add_argument('--tag-pattern', metavar='REGEX', help='A regular expression specifying which git tags to match.\n\nThis pattern is used as a regular expression search for any incoming pushes.\nFor example, --tag-pattern=foo will match "foo", "foobar", and "barfoo".\nEvents on a tag that does not match will be ignored.\n\nThe syntax of the regular expressions accepted is the syntax accepted by\nRE2 and described at https://github.com/google/re2/wiki/Syntax.\n')