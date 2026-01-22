from __future__ import absolute_import, division, print_function
import json
from importlib import import_module
from ansible.errors import AnsibleActionFail
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible.module_utils.connection import ConnectionError as AnsibleConnectionError
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.modules.cli_parse import DOCUMENTATION
def _load_parser(self, task_vars):
    """Load a parser from the fs

        :param task_vars: The vars provided when the task was run
        :type task_vars: dict
        :return: An instance of class CliParser
        :rtype: CliParser
        """
    requested_parser = self._task.args.get('parser').get('name')
    cref = dict(zip(['corg', 'cname', 'plugin'], requested_parser.split('.')))
    if cref['cname'] == 'netcommon' and cref['plugin'] in ['json', 'textfsm', 'ttp', 'xml']:
        cref['cname'] = 'utils'
        msg = "Use 'ansible.utils.{plugin}' for parser name instead of '{requested_parser}'. This feature will be removed from 'ansible.netcommon' collection in a release after 2022-11-01".format(plugin=cref['plugin'], requested_parser=requested_parser)
        self._display.warning(msg)
    parserlib = 'ansible_collections.{corg}.{cname}.plugins.sub_plugins.cli_parser.{plugin}_parser'.format(**cref)
    try:
        parsercls = getattr(import_module(parserlib), self.PARSER_CLS_NAME)
        parser = parsercls(task_args=self._task.args, task_vars=task_vars, debug=self._debug)
        return parser
    except Exception as exc:
        if cref['cname'] == 'netcommon' and cref['plugin'] in ['native', 'content_templates', 'ntc', 'pyats']:
            parserlib = 'ansible_collections.{corg}.{cname}.plugins.cli_parsers.{plugin}_parser'.format(**cref)
            try:
                parsercls = getattr(import_module(parserlib), self.PARSER_CLS_NAME)
                parser = parsercls(task_args=self._task.args, task_vars=task_vars, debug=self._debug)
                return parser
            except Exception as exc:
                self._result['failed'] = True
                self._result['msg'] = 'Error loading parser: {err}'.format(err=to_native(exc))
                return None
        self._result['failed'] = True
        self._result['msg'] = 'Error loading parser: {err}'.format(err=to_native(exc))
        return None