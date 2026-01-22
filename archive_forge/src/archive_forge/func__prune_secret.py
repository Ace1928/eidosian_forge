import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
def _prune_secret(action):
    if action['Type'] != 'authenticate-oidc':
        return action
    if not action['AuthenticateOidcConfig'].get('Scope', False):
        action['AuthenticateOidcConfig']['Scope'] = 'openid'
    if not action['AuthenticateOidcConfig'].get('SessionTimeout', False):
        action['AuthenticateOidcConfig']['SessionTimeout'] = 604800
    if action['AuthenticateOidcConfig'].get('UseExistingClientSecret', False):
        action['AuthenticateOidcConfig'].pop('ClientSecret', None)
    if not action['AuthenticateOidcConfig'].get('OnUnauthenticatedRequest', False):
        action['AuthenticateOidcConfig']['OnUnauthenticatedRequest'] = 'authenticate'
    if not action['AuthenticateOidcConfig'].get('SessionCookieName', False):
        action['AuthenticateOidcConfig']['SessionCookieName'] = 'AWSELBAuthSessionCookie'
    return action