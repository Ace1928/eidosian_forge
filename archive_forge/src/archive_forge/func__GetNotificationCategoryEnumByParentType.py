from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.essential_contacts import contacts
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.essential_contacts import flags
from googlecloudsdk.command_lib.essential_contacts import util
@staticmethod
def _GetNotificationCategoryEnumByParentType(parent_name):
    """Gets the NotificationCategory enum to cast the args as based on the type of parent resource arg."""
    if parent_name.startswith('folders'):
        return contacts.GetMessages().EssentialcontactsFoldersContactsComputeRequest.NotificationCategoriesValueValuesEnum
    if parent_name.startswith('organizations'):
        return contacts.GetMessages().EssentialcontactsOrganizationsContactsComputeRequest.NotificationCategoriesValueValuesEnum
    return contacts.GetMessages().EssentialcontactsProjectsContactsComputeRequest.NotificationCategoriesValueValuesEnum