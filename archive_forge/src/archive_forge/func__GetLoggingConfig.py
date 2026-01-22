from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def _GetLoggingConfig(options, messages):
    """Gets the LoggingConfig from create and update options."""
    if options.logging is None:
        return None
    if any((c not in LOGGING_OPTIONS for c in options.logging)):
        raise util.Error('[' + ', '.join(options.logging) + '] contains option(s) that are not supported for logging.')
    config = messages.LoggingComponentConfig()
    if NONE in options.logging:
        if len(options.logging) > 1:
            raise util.Error('Cannot include other values when None is specified.')
        return messages.LoggingConfig(componentConfig=config)
    if SYSTEM not in options.logging:
        raise util.Error('Must include system logging if any logging is enabled.')
    config.enableComponents.append(messages.LoggingComponentConfig.EnableComponentsValueListEntryValuesEnum.SYSTEM_COMPONENTS)
    if WORKLOAD in options.logging:
        config.enableComponents.append(messages.LoggingComponentConfig.EnableComponentsValueListEntryValuesEnum.WORKLOADS)
    if API_SERVER in options.logging:
        config.enableComponents.append(messages.LoggingComponentConfig.EnableComponentsValueListEntryValuesEnum.APISERVER)
    if SCHEDULER in options.logging:
        config.enableComponents.append(messages.LoggingComponentConfig.EnableComponentsValueListEntryValuesEnum.SCHEDULER)
    if CONTROLLER_MANAGER in options.logging:
        config.enableComponents.append(messages.LoggingComponentConfig.EnableComponentsValueListEntryValuesEnum.CONTROLLER_MANAGER)
    if ADDON_MANAGER in options.logging:
        config.enableComponents.append(messages.LoggingComponentConfig.EnableComponentsValueListEntryValuesEnum.ADDON_MANAGER)
    return messages.LoggingConfig(componentConfig=config)