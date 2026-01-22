from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkemulticloud import azure as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.azure import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import endpoint_util
from googlecloudsdk.command_lib.container.gkemulticloud import flags
from googlecloudsdk.core import log
def _GetCert(self, client):
    if client.pemCertificate:
        return client.pemCertificate
    client_dict = encoding.MessageToPyValue(client)
    if 'certificate' in client_dict:
        return base64.b64decode(client_dict['certificate'].encode('utf-8'))