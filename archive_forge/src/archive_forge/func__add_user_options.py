import copy
import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
def _add_user_options(parser):
    parser.add_argument('--ignore-lockout-failure-attempts', action='store_true', help=_('Opt into ignoring the number of times a user has authenticated and locking out the user as a result'))
    parser.add_argument('--no-ignore-lockout-failure-attempts', action='store_true', help=_('Opt out of ignoring the number of times a user has authenticated and locking out the user as a result'))
    parser.add_argument('--ignore-password-expiry', action='store_true', help=_('Opt into allowing user to continue using passwords that may be expired'))
    parser.add_argument('--no-ignore-password-expiry', action='store_true', help=_('Opt out of allowing user to continue using passwords that may be expired'))
    parser.add_argument('--ignore-change-password-upon-first-use', action='store_true', help=_('Control if a user should be forced to change their password immediately after they log into keystone for the first time. Opt into ignoring the user to change their password during first time login in keystone'))
    parser.add_argument('--no-ignore-change-password-upon-first-use', action='store_true', help=_('Control if a user should be forced to change their password immediately after they log into keystone for the first time. Opt out of ignoring the user to change their password during first time login in keystone'))
    parser.add_argument('--enable-lock-password', action='store_true', help=_('Disables the ability for a user to change its password through self-service APIs'))
    parser.add_argument('--disable-lock-password', action='store_true', help=_('Enables the ability for a user to change its password through self-service APIs'))
    parser.add_argument('--enable-multi-factor-auth', action='store_true', help=_('Enables the MFA (Multi Factor Auth)'))
    parser.add_argument('--disable-multi-factor-auth', action='store_true', help=_('Disables the MFA (Multi Factor Auth)'))
    parser.add_argument('--multi-factor-auth-rule', metavar='<rule>', action='append', default=[], help=_('Set multi-factor auth rules. For example, to set a rule requiring the "password" and "totp" auth methods to be provided, use: "--multi-factor-auth-rule password,totp". May be provided multiple times to set different rule combinations.'))