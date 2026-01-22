from .checks import check_data
from .specs import (
def _decode_data_bytes(status_byte, data, spec):
    if len(data) != spec['length'] - 1:
        raise ValueError('wrong number of bytes for {} message'.format(spec['type']))
    names = [name for name in spec['value_names'] if name != 'channel']
    args = {name: value for name, value in zip(names, data)}
    if status_byte in CHANNEL_MESSAGES:
        args['channel'] = status_byte & 15
    return args