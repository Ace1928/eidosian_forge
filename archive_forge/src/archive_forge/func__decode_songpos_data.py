from .checks import check_data
from .specs import (
def _decode_songpos_data(data):
    return {'pos': data[0] | data[1] << 7}