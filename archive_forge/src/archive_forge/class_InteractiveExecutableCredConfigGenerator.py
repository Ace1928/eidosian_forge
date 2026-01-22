from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class InteractiveExecutableCredConfigGenerator(ExecutableCredConfigGenerator):
    """The generator for executable-command-based credentials configs with interactive mode."""

    def __init__(self, config_type, command, timeout_millis, output_file, interactive_timeout_millis):
        super(InteractiveExecutableCredConfigGenerator, self).__init__(config_type, command, timeout_millis, output_file)
        self.interactive_timeout_millis = int(interactive_timeout_millis)

    def get_source(self, args):
        if not self.output_file:
            raise GeneratorError('--executable-output-file must be specified if ' + '--interactive-timeout-millis is provided.')
        executable_config = {'command': self.command, 'timeout_millis': self.timeout_millis, 'output_file': self.output_file, 'interactive_timeout_millis': self.interactive_timeout_millis}
        return {'executable': executable_config}