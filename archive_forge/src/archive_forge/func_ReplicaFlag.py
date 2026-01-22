from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from cloudsdk.google.protobuf import descriptor_pb2
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core.util import files
def ReplicaFlag(parser, name, text, required=True):
    return parser.add_argument(name, required=required, metavar='location=LOCATION,type=TYPE', action='store', type=arg_parsers.ArgList(custom_delim_char=':', min_length=1, element_type=arg_parsers.ArgDict(spec={'location': str, 'type': str}, required_keys=['location', 'type'])), help=text)