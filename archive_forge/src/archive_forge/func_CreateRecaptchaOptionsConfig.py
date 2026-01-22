from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateRecaptchaOptionsConfig(client, args, existing_recaptcha_options_config):
    """Returns a SecurityPolicyRecaptchaOptionsConfig message."""
    messages = client.messages
    recaptcha_options_config = existing_recaptcha_options_config if existing_recaptcha_options_config is not None else messages.SecurityPolicyRecaptchaOptionsConfig()
    if args.IsSpecified('recaptcha_redirect_site_key'):
        recaptcha_options_config.redirectSiteKey = args.recaptcha_redirect_site_key
    return recaptcha_options_config