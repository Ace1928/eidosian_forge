from .checks import check_data
from .specs import (
def _make_special_cases():
    cases = {224: _decode_pitchwheel_data, 240: _decode_sysex_data, 241: _decode_quarter_frame_data, 242: _decode_songpos_data}
    for i in range(16):
        cases[224 | i] = _decode_pitchwheel_data
    return cases