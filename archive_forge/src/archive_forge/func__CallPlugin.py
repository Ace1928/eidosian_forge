import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import baseauthenticator
def _CallPlugin(self, cmd, input_json):
    """Calls the plugin and validates the response."""
    input_length = len(input_json)
    length_bytes_le = struct.pack('<I', input_length)
    request = length_bytes_le + input_json.encode()
    sign_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout = sign_process.communicate(request)[0]
    exit_status = sign_process.wait()
    response_len_le = stdout[:4]
    response_len = struct.unpack('<I', response_len_le)[0]
    response = stdout[4:]
    if response_len != len(response):
        raise errors.PluginError('Plugin response length {} does not match data {} (exit_status={})'.format(response_len, len(response), exit_status))
    try:
        json_response = json.loads(response.decode())
    except ValueError:
        raise errors.PluginError('Plugin returned invalid output (exit_status={})'.format(exit_status))
    if json_response.get('type') != 'sign_helper_reply':
        raise errors.PluginError('Plugin returned invalid response type (exit_status={})'.format(exit_status))
    result_code = json_response.get('code')
    if result_code is None:
        raise errors.PluginError('Plugin missing result code (exit_status={})'.format(exit_status))
    if result_code == SK_SIGNING_PLUGIN_TOUCH_REQUIRED:
        raise errors.U2FError(errors.U2FError.TIMEOUT)
    elif result_code == SK_SIGNING_PLUGIN_WRONG_DATA:
        raise errors.U2FError(errors.U2FError.DEVICE_INELIGIBLE)
    elif result_code != SK_SIGNING_PLUGIN_NO_ERROR:
        raise errors.PluginError('Plugin failed with error {} - {} (exit_status={})'.format(result_code, json_response.get('errorDetail'), exit_status))
    response_data = json_response.get('responseData')
    if response_data is None:
        raise errors.PluginErrors('Plugin returned output with missing responseData (exit_status={})'.format(exit_status))
    return response_data