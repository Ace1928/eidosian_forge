from .specs import CHANNEL_MESSAGES, MIN_PITCHWHEEL, SPEC_BY_TYPE
def _encode_pitchwheel(msg):
    pitch = msg['pitch'] - MIN_PITCHWHEEL
    return [224 | msg['channel'], pitch & 127, pitch >> 7]