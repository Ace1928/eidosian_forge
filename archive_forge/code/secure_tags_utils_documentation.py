from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import tag_utils
Returns a unified secure tag identifier.

  Translates the namespaced tag if required.

  Args:
    secure_tag: secure tag value in format tagValues/ID or
      ORG_ID/TAG_KEY_NAME/TAG_VALUE_NAME

  Returns:
    Secure tag name in unified format tagValues/ID
  