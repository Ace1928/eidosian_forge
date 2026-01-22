from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
def _MakeEnumValue(self, enum):
    """Make an enum value."""
    return self.messages.GoogleCloudDatacatalogV1FieldTypeEnumTypeEnumValue(displayName=enum)