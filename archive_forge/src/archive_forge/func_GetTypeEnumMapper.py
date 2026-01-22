from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def GetTypeEnumMapper(version=None):
    return arg_utils.ChoiceEnumMapper('--type', util.GetMessages(version=version).ServicePerimeter.PerimeterTypeValueValuesEnum, custom_mappings={'PERIMETER_TYPE_REGULAR': 'regular', 'PERIMETER_TYPE_BRIDGE': 'bridge'}, required=False, help_str='          Type of the perimeter.\n\n          A *regular* perimeter allows resources within this service perimeter\n          to import and export data amongst themselves. A project may belong to\n          at most one regular service perimeter.\n\n          A *bridge* perimeter allows resources in different regular service\n          perimeters to import and export data between each other. A project may\n          belong to multiple bridge service perimeters (only if it also belongs to a\n          regular service perimeter). Both restricted and unrestricted service lists,\n          as well as access level lists, must be empty.\n          ')