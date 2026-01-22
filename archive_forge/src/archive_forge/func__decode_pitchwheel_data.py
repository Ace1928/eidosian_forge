from .checks import check_data
from .specs import (
def _decode_pitchwheel_data(data):
    return {'pitch': data[0] | (data[1] << 7) + MIN_PITCHWHEEL}