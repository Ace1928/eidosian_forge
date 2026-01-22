import hashlib
import hmac
import os
import struct
from ntlm_auth.compute_response import ComputeResponse
from ntlm_auth.constants import AvId, AvFlags, MessageTypes, NegotiateFlags, \
from ntlm_auth.rc4 import ARC4
def add_mic(self, negotiate_message, challenge_message):
    if self.target_info is not None:
        av_flags = self.target_info[AvId.MSV_AV_FLAGS]
        if av_flags is not None and av_flags == struct.pack('<L', AvFlags.MIC_PROVIDED):
            self.mic = struct.pack('<IIII', 0, 0, 0, 0)
            negotiate_data = negotiate_message.get_data()
            challenge_data = challenge_message.get_data()
            authenticate_data = self.get_data()
            hmac_data = negotiate_data + challenge_data + authenticate_data
            mic = hmac.new(self.exported_session_key, hmac_data, digestmod=hashlib.md5).digest()
            self.mic = mic