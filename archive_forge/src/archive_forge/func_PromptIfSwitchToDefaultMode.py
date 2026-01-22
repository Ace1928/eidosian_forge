from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def PromptIfSwitchToDefaultMode(messages, resource_class, existing_mode, new_mode):
    """If necessary, prompts the user for switching modes."""
    if existing_mode is not None and existing_mode is resource_class.AdvertiseModeValueValuesEnum.CUSTOM and (new_mode is not None) and (new_mode is resource_class.AdvertiseModeValueValuesEnum.DEFAULT):
        resource_str = _GetResourceClassStr(messages, resource_class)
        console_io.PromptContinue(message=_MODE_SWITCH_MESSAGE.format(resource=resource_str), cancel_on_no=True)