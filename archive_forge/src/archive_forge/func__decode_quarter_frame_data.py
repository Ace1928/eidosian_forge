from .checks import check_data
from .specs import (
def _decode_quarter_frame_data(data):
    return {'frame_type': data[0] >> 4, 'frame_value': data[0] & 15}