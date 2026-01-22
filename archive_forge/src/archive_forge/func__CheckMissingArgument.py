from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import target_proxies_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.certificate_manager import resource_args
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.backend_services import (
from googlecloudsdk.command_lib.compute.ssl_certificates import (
from googlecloudsdk.command_lib.compute.ssl_policies import (flags as
from googlecloudsdk.command_lib.compute.target_ssl_proxies import flags
from googlecloudsdk.command_lib.compute.target_ssl_proxies import target_ssl_proxies_utils
def _CheckMissingArgument(self, args):
    """Checks for missing argument."""
    all_args = ['ssl_certificates', 'proxy_header', 'backend_service', 'ssl_policy', 'clear_ssl_policy']
    err_msg_args = ['[--ssl-certificates]', '[--backend-service]', '[--proxy-header]', '[--ssl-policy]', '[--clear-ssl-policy]']
    if self._certificate_map:
        all_args.append('certificate_map')
        err_msg_args.append('[--certificate-map]')
        all_args.append('clear_certificate_map')
        err_msg_args.append('[--clear-certificate-map]')
        all_args.append('clear_ssl_certificates')
        err_msg_args.append('[--clear-ssl-certificates]')
    if not sum((args.IsSpecified(arg) for arg in all_args)):
        raise compute_exceptions.UpdatePropertyError('You must specify at least one of %s or %s.' % (', '.join(err_msg_args[:-1]), err_msg_args[-1]))