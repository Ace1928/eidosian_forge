from __future__ import absolute_import, division, print_function
import re
from collections import deque
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.acls.acls import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _parse_match_options(rendered_ace, ace_queue):
    """
            Parses the ACE queue and populates remaining options in the ACE dictionary,
            i.e., `rendered_ace`

            :param rendered_ace: The dictionary containing the ACE in structured format
            :param ace_queue: The ACE queue
            """
    if len(ace_queue) > 0:
        copy_ace_queue = deepcopy(ace_queue)
        for element in copy_ace_queue:
            if element == 'precedence':
                ace_queue.popleft()
                rendered_ace['precedence'] = ace_queue.popleft()
            elif element == 'dscp':
                ace_queue.popleft()
                dscp = {}
                operation = ace_queue.popleft()
                if operation in ('eq', 'gt', 'neq', 'lt', 'range'):
                    if operation == 'range':
                        dscp['range'] = {'start': ace_queue.popleft(), 'end': ace_queue.popleft()}
                    else:
                        dscp[operation] = ace_queue.popleft()
                else:
                    dscp['eq'] = operation
                rendered_ace['dscp'] = dscp
            elif element in ('packet-length', 'ttl'):
                ace_queue.popleft()
                element_dict = {}
                operation = ace_queue.popleft()
                if operation == 'range':
                    element_dict['range'] = {'start': ace_queue.popleft(), 'end': ace_queue.popleft()}
                else:
                    element_dict[operation] = ace_queue.popleft()
                rendered_ace[element.replace('-', '_')] = element_dict
            elif element in ('log', 'log-input', 'fragments', 'icmp-off', 'capture', 'destopts', 'authen', 'routing', 'hop-by-hop'):
                rendered_ace[element.replace('-', '_')] = True
                ace_queue.remove(element)
            copy_ace_queue = deepcopy(ace_queue)