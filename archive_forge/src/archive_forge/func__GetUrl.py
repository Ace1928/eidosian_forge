from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.config import config_helper
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import proxy
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.credentials import store
def _GetUrl(self, serv, tag, serv_id):
    if not serv.status:
        raise exceptions.ArgumentError('Status of service [{}] is not ready'.format(serv_id))
    if tag:
        for t in serv.status.traffic:
            if t.tag == tag:
                if not t.url:
                    raise exceptions.ArgumentError('URL for tag [{}] in service [{}] is not ready'.format(tag, serv_id))
                return t.url
        raise exceptions.ArgumentError('Cannot find tag [{}] in service [{}].'.format(tag, serv_id))
    if not serv.status.url:
        raise exceptions.ArgumentError('URL for service [{}] is not ready'.format(serv_id))
    return serv.status.url