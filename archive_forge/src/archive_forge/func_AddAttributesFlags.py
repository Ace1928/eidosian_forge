from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apphub import utils as apphub_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddAttributesFlags(parser, resource_name='application', release_track=base.ReleaseTrack.ALPHA):
    """Adds flags required for attributes fields."""
    parser.add_argument('--criticality-type', choices={'TYPE_UNSPECIFIED': 'Unspecified criticality type', 'MISSION_CRITICAL': 'Mission critical service, application or workload', 'HIGH': 'High impact', 'MEDIUM': 'Medium impact', 'LOW': 'Low impact'}, help='Criticality Type of the {}'.format(resource_name))
    parser.add_argument('--environment-type', choices={'TYPE_UNSPECIFIED': 'Unspecified environment type', 'PRODUCTION': 'Production environment', 'STAGING': 'Staging environment', 'TEST': 'Test environment', 'DEVELOPMENT': 'Development environment'}, help='Environment Type of the {}'.format(resource_name))
    if release_track == base.ReleaseTrack.ALPHA:
        parser.add_argument('--business-owners', type=arg_parsers.ArgDict(spec={'display-name': str, 'email': str, 'channel-uri': str}, required_keys=['email']), action='append', help='Business owners of the {}'.format(resource_name))
        parser.add_argument('--developer-owners', type=arg_parsers.ArgDict(spec={'display-name': str, 'email': str, 'channel-uri': str}, required_keys=['email']), action='append', help='Developer owners of the {}'.format(resource_name))
        parser.add_argument('--operator-owners', type=arg_parsers.ArgDict(spec={'display-name': str, 'email': str, 'channel-uri': str}, required_keys=['email']), action='append', help='Operator owners of the {}'.format(resource_name))
    elif release_track == base.ReleaseTrack.GA:
        parser.add_argument('--business-owners', type=arg_parsers.ArgDict(spec={'display-name': str, 'email': str}, required_keys=['email']), action='append', help='Business owners of the {}'.format(resource_name))
        parser.add_argument('--developer-owners', type=arg_parsers.ArgDict(spec={'display-name': str, 'email': str}, required_keys=['email']), action='append', help='Developer owners of the {}'.format(resource_name))
        parser.add_argument('--operator-owners', type=arg_parsers.ArgDict(spec={'display-name': str, 'email': str}, required_keys=['email']), action='append', help='Operator owners of the {}'.format(resource_name))