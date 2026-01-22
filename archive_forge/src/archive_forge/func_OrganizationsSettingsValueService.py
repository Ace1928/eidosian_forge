from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def OrganizationsSettingsValueService():
    """Returns the service class for the Organization Settings Value resource."""
    client = ResourceSettingsClient()
    return client.organizations_settings_value