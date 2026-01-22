import argparse
import collections
import getpass
import logging
import sys
from urllib import parse as urlparse
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
from keystoneauth1 import session
from oslo_utils import importutils
import requests
import cinderclient
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import client
from cinderclient import exceptions as exc
from cinderclient import utils
def _append_global_identity_args(self, parser):
    loading.register_session_argparse_arguments(parser)
    default_auth_plugin = 'password'
    loading.register_auth_argparse_arguments(parser, [], default=default_auth_plugin)
    parser.add_argument('--os-auth-strategy', metavar='<auth-strategy>', default=utils.env('OS_AUTH_STRATEGY', default='keystone'), help=_('Authentication strategy (Env: OS_AUTH_STRATEGY, default keystone). For now, any other value will disable the authentication.'))
    parser.add_argument('--os_auth_strategy', help=argparse.SUPPRESS)
    env_plugin = utils.env('OS_AUTH_TYPE', 'OS_AUTH_PLUGIN', 'OS_AUTH_SYSTEM')
    parser.set_defaults(os_auth_type=env_plugin)
    parser.add_argument('--os_auth_type', help=argparse.SUPPRESS)
    parser.set_defaults(os_username=utils.env('OS_USERNAME', 'CINDER_USERNAME'))
    parser.add_argument('--os_username', help=argparse.SUPPRESS)
    parser.set_defaults(os_password=utils.env('OS_PASSWORD', 'CINDER_PASSWORD'))
    parser.add_argument('--os_password', help=argparse.SUPPRESS)
    parser.set_defaults(os_project_name=utils.env('OS_PROJECT_NAME', 'CINDER_PROJECT_ID'))
    parser.add_argument('--os_project_name', help=argparse.SUPPRESS)
    parser.set_defaults(os_project_id=utils.env('OS_PROJECT_ID', 'CINDER_PROJECT_ID'))
    parser.add_argument('--os_project_id', help=argparse.SUPPRESS)
    parser.set_defaults(os_auth_url=utils.env('OS_AUTH_URL', 'CINDER_URL'))
    parser.add_argument('--os_auth_url', help=argparse.SUPPRESS)
    parser.set_defaults(os_user_id=utils.env('OS_USER_ID'))
    parser.add_argument('--os_user_id', help=argparse.SUPPRESS)
    parser.set_defaults(os_user_domain_id=utils.env('OS_USER_DOMAIN_ID'))
    parser.add_argument('--os_user_domain_id', help=argparse.SUPPRESS)
    parser.set_defaults(os_user_domain_name=utils.env('OS_USER_DOMAIN_NAME'))
    parser.add_argument('--os_user_domain_name', help=argparse.SUPPRESS)
    parser.set_defaults(os_project_domain_id=utils.env('OS_PROJECT_DOMAIN_ID'))
    parser.set_defaults(os_project_domain_name=utils.env('OS_PROJECT_DOMAIN_NAME'))
    parser.set_defaults(os_region_name=utils.env('OS_REGION_NAME', 'CINDER_REGION_NAME'))
    parser.add_argument('--os_region_name', help=argparse.SUPPRESS)
    parser.add_argument('--os-token', metavar='<token>', default=utils.env('OS_TOKEN'), help=_('Defaults to env[OS_TOKEN].'))
    parser.add_argument('--os_token', help=argparse.SUPPRESS)
    parser.add_argument('--os-url', metavar='<url>', default=utils.env('OS_URL'), help=_('Defaults to env[OS_URL].'))
    parser.add_argument('--os_url', help=argparse.SUPPRESS)
    parser.set_defaults(insecure=utils.env('CINDERCLIENT_INSECURE', default=False))