from .specs import CHANNEL_MESSAGES, MIN_PITCHWHEEL, SPEC_BY_TYPE
def _encode_control_change(msg):
    return [176 | msg['channel'], msg['control'], msg['value']]