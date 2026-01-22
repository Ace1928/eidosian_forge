from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
class ListCommandParameterInfo(parameter_info_lib.ParameterInfoByConvention):

    def GetFlag(self, parameter_name, parameter_value=None, check_properties=True, for_update=False):
        return super(ListCommandParameterInfo, self).GetFlag(parameter_name, parameter_value=parameter_value, check_properties=check_properties, for_update=for_update)