from __future__ import absolute_import, division, print_function
import re
from collections import deque
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.acls.acls import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _parse_src_dest(rendered_ace, ace_queue, direction):
    """
            Parses the ACE queue and populates address, wildcard_bits,
            host or any keys in the source/destination dictionary of
            ace dictionary, i.e., `rendered_ace`.

            :param rendered_ace: The dictionary containing the ACE in structured format
            :param ace_queue: The ACE queue
            :param direction: Specifies whether to populate `source` or `destination`
                              dictionary
            """
    element = ace_queue.popleft()
    if element == 'host':
        rendered_ace[direction] = {'host': ace_queue.popleft()}
    elif element == 'net-group':
        rendered_ace[direction] = {'net_group': ace_queue.popleft()}
    elif element == 'port-group':
        rendered_ace[direction] = {'port_group': ace_queue.popleft()}
    elif element == 'any':
        rendered_ace[direction] = {'any': True}
    elif '/' in element:
        rendered_ace[direction] = {'prefix': element}
    elif isipaddress(to_text(element)):
        rendered_ace[direction] = {'address': element, 'wildcard_bits': ace_queue.popleft()}