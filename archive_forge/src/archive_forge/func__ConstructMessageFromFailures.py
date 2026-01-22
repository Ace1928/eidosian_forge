from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.diagnostics import check_base
from googlecloudsdk.core.diagnostics import diagnostic_base
import six
def _ConstructMessageFromFailures(self, failures, first_run):
    message = 'Hidden Property Check {0}.\n'.format('failed' if first_run else 'still does not pass')
    if failures:
        message += 'The following hidden properties have been set:\n'
    for failure in failures:
        message += '    {0}\n'.format(failure.message)
    if first_run:
        message += 'Properties files\n    User: {0}\n    Installation: {1}\n'.format(named_configs.ConfigurationStore.ActiveConfig().file_path, config.Paths().installation_properties_path)
    return message