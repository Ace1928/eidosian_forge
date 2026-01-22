from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.certificate_manager import api_client
Updates a certificate map entry.

    Used for updating labels, description and attached certificates.

    Args:
      entry_ref: a Resource reference to a
        certificatemanager.projects.locations.certificateMaps.certificateMapEntries
        resource.
      labels: unified GCP Labels for the resource.
      description: str, new description
      cert_refs: Resource references to
        certificatemanager.projects.locations.certificates resources to be
        attached to this entry.

    Returns:
      Operation: the long running operation to patch certificate map entry.
    