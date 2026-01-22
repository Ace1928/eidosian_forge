from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import stat
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError
from ansible.executor.playbook_executor import PlaybookExecutor
from ansible.module_utils.common.text.converters import to_bytes
from ansible.playbook.block import Block
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.utils.display import Display
def _process_block(b):
    taskmsg = ''
    for task in b.block:
        if isinstance(task, Block):
            taskmsg += _process_block(task)
        else:
            if task.action in C._ACTION_META and task.implicit:
                continue
            all_tags.update(task.tags)
            if context.CLIARGS['listtasks']:
                cur_tags = list(mytags.union(set(task.tags)))
                cur_tags.sort()
                if task.name:
                    taskmsg += '      %s' % task.get_name()
                else:
                    taskmsg += '      %s' % task.action
                taskmsg += '\tTAGS: [%s]\n' % ', '.join(cur_tags)
    return taskmsg