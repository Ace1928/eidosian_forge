from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class CloudResource(Resource):
    """For Resource classes with CloudUrl's.

  Attributes:
    TYPE_STRING (str): String representing the resource's content type.
    scheme (storage_url.ProviderPrefix): Prefix indicating what cloud provider
        hosts the bucket.
    storage_url (StorageUrl): A StorageUrl object representing the resource.
  """
    TYPE_STRING = 'cloud_resource'

    @property
    def scheme(self):
        return self.storage_url.scheme

    def get_formatted_acl(self):
        """Returns provider specific formatting for the acl fields.

    Provider specific resource classses can override this method to return
    provider specific formatting for acl fields. If not overriden, acl values
    are displayed as-is if present.

    Returns:
      Dictionary with acl fields as key and corresponding formatted values.
    """
        return {}