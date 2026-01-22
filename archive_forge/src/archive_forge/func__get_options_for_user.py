import copy
import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
def _get_options_for_user(identity_client, parsed_args):
    options = {}
    if parsed_args.ignore_lockout_failure_attempts:
        options['ignore_lockout_failure_attempts'] = True
    if parsed_args.no_ignore_lockout_failure_attempts:
        options['ignore_lockout_failure_attempts'] = False
    if parsed_args.ignore_password_expiry:
        options['ignore_password_expiry'] = True
    if parsed_args.no_ignore_password_expiry:
        options['ignore_password_expiry'] = False
    if parsed_args.ignore_change_password_upon_first_use:
        options['ignore_change_password_upon_first_use'] = True
    if parsed_args.no_ignore_change_password_upon_first_use:
        options['ignore_change_password_upon_first_use'] = False
    if parsed_args.enable_lock_password:
        options['lock_password'] = True
    if parsed_args.disable_lock_password:
        options['lock_password'] = False
    if parsed_args.enable_multi_factor_auth:
        options['multi_factor_auth_enabled'] = True
    if parsed_args.disable_multi_factor_auth:
        options['multi_factor_auth_enabled'] = False
    if parsed_args.multi_factor_auth_rule:
        auth_rules = [rule.split(',') for rule in parsed_args.multi_factor_auth_rule]
        if auth_rules:
            options['multi_factor_auth_rules'] = auth_rules
    return options