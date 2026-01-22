from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam.byoid_utilities import cred_config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
class CreateLoginConfig(base.CreateCommand):
    """Create a login configuration file to enable sign-in via a web-based authorization flow using Workforce Identity Federation.

  This command creates a configuration file to enable browser based
  third-party sign in with Workforce Identity Federation through
  `gcloud auth login --login-config=/path/to/config.json`.
  """
    detailed_help = {'EXAMPLES': textwrap.dedent('          To create a login configuration for your project, run:\n\n            $ {command} locations/global/workforcePools/$WORKFORCE_POOL_ID/providers/$PROVIDER_ID --output-file=login-config.json\n          ')}

    @classmethod
    def Args(cls, parser):
        parser.add_argument('audience', help='Workforce pool provider resource name.')
        parser.add_argument('--output-file', help='Location to store the generated login configuration file.', required=True)
        parser.add_argument('--activate', action='store_true', default=False, help='Sets the property `auth/login_config_file` to the created login configuration file. Calling `gcloud auth login` will automatically use this login configuration unless it is explicitly unset.')
        parser.add_argument('--enable-mtls', help='Use mTLS for STS endpoints.', action='store_true', hidden=True)
        parser.add_argument('--universe-domain', help='The universe domain.', hidden=True)
        parser.add_argument('--universe-cloud-web-domain', help='The universe cloud web domain.', hidden=True)

    def Run(self, args):
        universe_domain_property = properties.VALUES.core.universe_domain
        universe_domain = universe_domain_property.Get()
        if getattr(args, 'universe_domain', None):
            universe_domain = args.universe_domain
        universe_cloud_web_domain = GOOGLE_DEFAULT_CLOUD_WEB_DOMAIN
        if getattr(args, 'universe_cloud_web_domain', None):
            universe_cloud_web_domain = args.universe_cloud_web_domain
        if universe_domain != universe_domain_property.default:
            if universe_cloud_web_domain == GOOGLE_DEFAULT_CLOUD_WEB_DOMAIN:
                raise exceptions.RequiredArgumentException('--universe-cloud-web-domain', 'The universe domain and universe cloud web domain must be specified together.')
        elif universe_cloud_web_domain != GOOGLE_DEFAULT_CLOUD_WEB_DOMAIN:
            raise exceptions.RequiredArgumentException('--universe-domain', 'The universe domain must be configured when the universe cloud web domain is specified.')
        enable_mtls = getattr(args, 'enable_mtls', False)
        token_endpoint_builder = cred_config.StsEndpoints(enable_mtls=enable_mtls, universe_domain=universe_domain)
        output = {'universe_domain': universe_domain, 'type': 'external_account_authorized_user_login_config', 'audience': '//iam.googleapis.com/' + args.audience, 'auth_url': 'https://auth.{cloud_web_domain}/authorize'.format(cloud_web_domain=universe_cloud_web_domain), 'token_url': token_endpoint_builder.oauth_token_url, 'token_info_url': token_endpoint_builder.token_info_url}
        if universe_cloud_web_domain != GOOGLE_DEFAULT_CLOUD_WEB_DOMAIN:
            output['universe_cloud_web_domain'] = universe_cloud_web_domain
        files.WriteFileContents(args.output_file, json.dumps(output, indent=2))
        log.CreatedResource(args.output_file, RESOURCE_TYPE)
        if args.activate:
            properties.PersistProperty(properties.VALUES.auth.login_config_file, os.path.abspath(args.output_file))