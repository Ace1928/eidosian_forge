from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test.ios import catalog_manager
from googlecloudsdk.calliope import exceptions
def AllArgsSet():
    """Returns a set containing the names of every iOS test arg."""
    return arg_util.GetSetOfAllTestArgs(TypedArgRules(), SharedArgRules())