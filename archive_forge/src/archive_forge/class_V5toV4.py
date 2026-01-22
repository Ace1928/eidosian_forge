import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
class V5toV4(Adapter):
    """Adapt msg protocol v5 to v4"""
    version = '4.1'
    msg_type_map = {'execute_result': 'pyout', 'execute_input': 'pyin', 'error': 'pyerr', 'inspect_request': 'object_info_request', 'inspect_reply': 'object_info_reply'}

    def update_header(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Update the header."""
        msg['header'].pop('version', None)
        msg['parent_header'].pop('version', None)
        return msg

    def kernel_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a kernel info reply."""
        v4c = {}
        content = msg['content']
        for key in ('language_version', 'protocol_version'):
            if key in content:
                v4c[key] = _version_str_to_list(content[key])
        if content.get('implementation', '') == 'ipython' and 'implementation_version' in content:
            v4c['ipython_version'] = _version_str_to_list(content['implementation_version'])
        language_info = content.get('language_info', {})
        language = language_info.get('name', '')
        v4c.setdefault('language', language)
        if 'version' in language_info:
            v4c.setdefault('language_version', _version_str_to_list(language_info['version']))
        msg['content'] = v4c
        return msg

    def execute_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute request."""
        content = msg['content']
        content.setdefault('user_variables', [])
        return msg

    def execute_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute reply."""
        content = msg['content']
        content.setdefault('user_variables', {})
        return msg

    def complete_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete request."""
        content = msg['content']
        code = content['code']
        cursor_pos = content['cursor_pos']
        line, cursor_pos = code_to_line(code, cursor_pos)
        new_content = msg['content'] = {}
        new_content['text'] = ''
        new_content['line'] = line
        new_content['block'] = None
        new_content['cursor_pos'] = cursor_pos
        return msg

    def complete_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete reply."""
        content = msg['content']
        cursor_start = content.pop('cursor_start')
        cursor_end = content.pop('cursor_end')
        match_len = cursor_end - cursor_start
        content['matched_text'] = content['matches'][0][:match_len]
        content.pop('metadata', None)
        return msg

    def object_info_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an object info request."""
        content = msg['content']
        code = content['code']
        cursor_pos = content['cursor_pos']
        line, _ = code_to_line(code, cursor_pos)
        new_content = msg['content'] = {}
        new_content['oname'] = extract_oname_v4(code, cursor_pos)
        new_content['detail_level'] = content['detail_level']
        return msg

    def object_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """inspect_reply can't be easily backward compatible"""
        msg['content'] = {'found': False, 'oname': 'unknown'}
        return msg

    def stream(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a stream message."""
        content = msg['content']
        content['data'] = content.pop('text')
        return msg

    def display_data(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a display data message."""
        content = msg['content']
        content.setdefault('source', 'display')
        data = content['data']
        if 'application/json' in data:
            try:
                data['application/json'] = json.dumps(data['application/json'])
            except Exception:
                pass
        return msg

    def input_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an input request."""
        msg['content'].pop('password', None)
        return msg