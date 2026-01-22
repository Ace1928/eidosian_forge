from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.executor.task_queue_manager import TaskQueueManager
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.splitter import parse_kv
from ansible.parsing.utils.yaml import from_yaml
from ansible.playbook import Playbook
from ansible.playbook.play import Play
from ansible.utils.display import Display
 create and execute the single task playbook 