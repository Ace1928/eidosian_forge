from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
Executes the json-rpc and returns the output received
        from remote device.
        :name: rpc method to be executed over connection plugin that implements jsonrpc 2.0
        :args: Ordered list of params passed as arguments to rpc method
        :kwargs: Dict of valid key, value pairs passed as arguments to rpc method

        For usage refer the respective connection plugin docs.
        