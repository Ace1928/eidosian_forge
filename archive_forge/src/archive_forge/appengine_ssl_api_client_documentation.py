from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
Updates a certificate for the given application.

    One of display_name, cert_path, or private_key_path should be set. Omitted
    fields will not be updated from their current value. Any invalid arguments
    will fail the entire command.

    Args:
      cert_id: str, the id of the certificate to update.
      display_name: str, the display name for a new certificate.
      cert_path: str, location on disk to a certificate file.
      private_key_path: str, location on disk to a private key file.

    Returns:
      The created AuthorizedCertificate object.

    Raises: InvalidInputError if the user does not specify both cert and key.
    