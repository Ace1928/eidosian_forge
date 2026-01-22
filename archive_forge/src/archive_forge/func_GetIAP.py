from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetIAP(iap_arg, messages, existing_iap_settings=None):
    """Returns IAP settings from arguments."""
    subargs = iap_arg.split(',')
    iap_arg_parsed = {}
    for subarg in subargs:
        if not subarg:
            continue
        if '=' in subarg:
            subarg, value = subarg.split('=', 1)
        else:
            value = True

        def _Repr(s):
            r = repr(s)
            if r.startswith('u'):
                r = r[1:]
            return r
        if subarg in ('enabled', 'disabled', 'oauth2-client-id', 'oauth2-client-secret'):
            if subarg in iap_arg_parsed:
                raise exceptions.InvalidArgumentException('--iap', 'Sub-argument %s specified multiple times' % _Repr(subarg))
            iap_arg_parsed[subarg] = value
        else:
            raise exceptions.InvalidArgumentException('--iap', 'Invalid sub-argument %s' % _Repr(subarg))
    if not iap_arg_parsed or not iap_arg:
        raise exceptions.InvalidArgumentException('--iap', 'Must provide value when specifying --iap')
    if 'enabled' in iap_arg_parsed and 'disabled' in iap_arg_parsed:
        raise exceptions.InvalidArgumentException('--iap', 'Must specify only one of [enabled] or [disabled]')
    iap_settings = messages.BackendServiceIAP()
    if 'enabled' in iap_arg_parsed:
        iap_settings.enabled = True
    elif 'disabled' in iap_arg_parsed:
        iap_settings.enabled = False
    elif existing_iap_settings is None:
        iap_settings.enabled = False
    else:
        iap_settings.enabled = existing_iap_settings.enabled
    if 'oauth2-client-id' in iap_arg_parsed or 'oauth2-client-secret' in iap_arg_parsed:
        iap_settings.oauth2ClientId = iap_arg_parsed.get('oauth2-client-id')
        iap_settings.oauth2ClientSecret = iap_arg_parsed.get('oauth2-client-secret')
        if not iap_settings.oauth2ClientId or not iap_settings.oauth2ClientSecret:
            raise exceptions.InvalidArgumentException('--iap', 'Both [oauth2-client-id] and [oauth2-client-secret] must be specified together')
    return iap_settings