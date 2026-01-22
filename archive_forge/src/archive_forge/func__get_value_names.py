import re
from .checks import check_data, check_msgdict, check_value
from .decode import decode_message
from .encode import encode_message
from .specs import REALTIME_TYPES, SPEC_BY_TYPE, make_msgdict
from .strings import msg2str, str2msg
def _get_value_names(self):
    return list(SPEC_BY_TYPE[self.type]['value_names']) + ['time']