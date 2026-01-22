from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from functools import wraps
from ansible.plugins import AnsiblePlugin
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_bytes, to_text
def _update_cli_prompt_context(self, config_context=None, exit_command='exit'):
    """
        Update the cli prompt context to ensure it is in operational mode
        :param config_context: It is string value to identify if the current cli prompt ends with config mode prompt
        :param exit_command: Command to execute to exit the config mode
        :return: None
        """
    out = self._connection.get_prompt()
    if out is None:
        raise AnsibleConnectionFailure(message=u'cli prompt is not identified from the last received response window: %s' % self._connection._last_recv_window)
    while True:
        out = to_text(out, errors='surrogate_then_replace').strip()
        if config_context and out.endswith(config_context):
            self._connection.queue_message('vvvv', 'wrong context, sending exit to device')
            self.send_command(exit_command)
            out = self._connection.get_prompt()
        else:
            break