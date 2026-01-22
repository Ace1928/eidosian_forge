from __future__ import (absolute_import, division, print_function)
import datetime
import json
import copy
from functools import partial
from ansible.inventory.host import Host
from ansible.module_utils._text import to_text
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def _write_event(self, event_name, output):
    output['_event'] = event_name
    output['_timestamp'] = current_time()
    self._display.display(json.dumps(output, cls=AnsibleJSONEncoder, indent=self._json_indent, separators=',:', sort_keys=True))