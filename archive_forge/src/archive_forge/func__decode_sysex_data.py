from .checks import check_data
from .specs import (
def _decode_sysex_data(data):
    return {'data': tuple(data)}