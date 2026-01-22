import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
class continuous_frame(object):

    def __init__(self, fire_cont_frame, skip_utf8_validation):
        self.fire_cont_frame = fire_cont_frame
        self.skip_utf8_validation = skip_utf8_validation
        self.cont_data = None
        self.recving_frames = None

    def validate(self, frame):
        if not self.recving_frames and frame.opcode == ABNF.OPCODE_CONT:
            raise WebSocketProtocolException('Illegal frame')
        if self.recving_frames and frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
            raise WebSocketProtocolException('Illegal frame')

    def add(self, frame):
        if self.cont_data:
            self.cont_data[1] += frame.data
        else:
            if frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
                self.recving_frames = frame.opcode
            self.cont_data = [frame.opcode, frame.data]
        if frame.fin:
            self.recving_frames = None

    def is_fire(self, frame):
        return frame.fin or self.fire_cont_frame

    def extract(self, frame):
        data = self.cont_data
        self.cont_data = None
        frame.data = data[1]
        if not self.fire_cont_frame and data[0] == ABNF.OPCODE_TEXT and (not self.skip_utf8_validation) and (not validate_utf8(frame.data)):
            raise WebSocketPayloadException('cannot decode: ' + repr(frame.data))
        return [data[0], frame]