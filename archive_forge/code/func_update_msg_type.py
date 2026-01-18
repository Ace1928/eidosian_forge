import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def update_msg_type(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Update the message type."""
    header = msg['header']
    msg_type = header['msg_type']
    if msg_type in self.msg_type_map:
        msg['msg_type'] = header['msg_type'] = self.msg_type_map[msg_type]
    return msg