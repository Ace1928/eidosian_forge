from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible import context
from ansible.executor.task_queue_manager import TaskQueueManager, AnsibleEndPlay
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.loader import become_loader, connection_loader, shell_loader
from ansible.playbook import Playbook
from ansible.template import Templar
from ansible.utils.helpers import pct_to_int
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.utils.path import makedirs_safe
from ansible.utils.ssh_functions import set_default_transport
from ansible.utils.display import Display
def _get_serialized_batches(self, play):
    """
        Returns a list of hosts, subdivided into batches based on
        the serial size specified in the play.
        """
    all_hosts = self._inventory.get_hosts(play.hosts, order=play.order)
    all_hosts_len = len(all_hosts)
    serial_batch_list = play.serial
    if len(serial_batch_list) == 0:
        serial_batch_list = [-1]
    cur_item = 0
    serialized_batches = []
    while len(all_hosts) > 0:
        serial = pct_to_int(serial_batch_list[cur_item], all_hosts_len)
        if serial <= 0:
            serialized_batches.append(all_hosts)
            break
        else:
            play_hosts = []
            for x in range(serial):
                if len(all_hosts) > 0:
                    play_hosts.append(all_hosts.pop(0))
            serialized_batches.append(play_hosts)
        cur_item += 1
        if cur_item > len(serial_batch_list) - 1:
            cur_item = len(serial_batch_list) - 1
    return serialized_batches