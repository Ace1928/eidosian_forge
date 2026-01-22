from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def ValidateComposerVersionExclusiveOptionFactory(composer_v1_option, error_message):
    """Creates Composer version specific ActionClass decorators."""

    def ValidateComposerVersionExclusiveOptionDecorator(action_class):
        """Decorates ActionClass to cross-validate argument with Composer version."""
        original_call = action_class.__call__

        def DecoratedCall(self, parser, namespace, value, option_string=None):

            def IsImageVersionStringComposerV1(image_version):
                return image_version.startswith('composer-1.') or image_version.startswith('composer-1-')
            try:
                if namespace.image_version and IsImageVersionStringComposerV1(namespace.image_version) != composer_v1_option:
                    raise command_util.InvalidUserInputError(error_message.format(opt=option_string))
            except AttributeError:
                pass
            original_call(self, parser, namespace, value, option_string)
        action_class.__call__ = DecoratedCall
        return action_class
    return ValidateComposerVersionExclusiveOptionDecorator