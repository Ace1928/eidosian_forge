from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.config import config_helper
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ConfigurationHelper(base.Command):
    """A helper for providing auth and config data to external tools."""
    detailed_help = {'DESCRIPTION': "            {description}\n\n            Tools can call out to this command to get gcloud's current auth and\n            configuration context when needed. This is appropriate when external\n            tools want to operate within the context of the user's current\n            gcloud session.\n\n            This command returns a nested data structure with the following\n            schema:\n\n            *  credential\n               *  access_token - string, The current OAuth2 access token\n               *  token_expiry - string, The time the token will expire. This\n                  can be empty for some credential types. It is a UTC time\n                  formatted as: '%Y-%m-%dT%H:%M:%SZ'\n            *  configuration\n               *  active_configuration - string, The name of the active gcloud\n                  configuration\n               *  properties - {string: {string: string}}, The full set of\n                  active gcloud properties\n        ", 'EXAMPLES': '            This command should always be used with the --format flag to get the\n            output in a structured format.\n\n            To get the current gcloud context:\n\n              $ {command} --format=json\n\n            To get the current gcloud context after forcing a refresh of the\n            OAuth2 credentials:\n\n              $ {command} --format=json --force-auth-refresh\n\n            To set MIN_EXPIRY amount of time that if given, refresh the\n            credentials if they are within MIN_EXPIRY from expiration:\n\n              $ {command} --format=json --min-expiry=MIN_EXPIRY\n\n            By default, MIN_EXPIRY is set to be 0 second.\n        '}

    @staticmethod
    def Args(parser):
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--force-auth-refresh', action='store_true', help='Force a refresh of the credentials even if they have not yet expired. By default, credentials will only refreshed when necessary.')
        group.add_argument('--min-expiry', type=arg_parsers.Duration(lower_bound='0s', upper_bound='1h'), help='If given, refresh the credentials if they are within MIN_EXPIRY from expiration.', default='0s')

    def Run(self, args):
        cred = store.Load(use_google_auth=True)
        if args.force_auth_refresh:
            store.Refresh(cred)
        else:
            store.RefreshIfExpireWithinWindow(cred, '{}'.format(args.min_expiry))
        config_name = named_configs.ConfigurationStore.ActiveConfig().name
        props = properties.VALUES.AllValues()
        return config_helper.ConfigHelperResult(credential=cred, active_configuration=config_name, properties=props)