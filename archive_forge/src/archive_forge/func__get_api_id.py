import threading
import rtmidi
from .. import ports
from ..messages import Message
from ._parser_queue import ParserQueue
from .rtmidi_utils import expand_alsa_port_name
def _get_api_id(name=None):
    if name is None:
        return rtmidi.API_UNSPECIFIED
    try:
        api = _name_to_api[name]
    except KeyError as ke:
        raise ValueError(f'unknown API {name}') from ke
    if name in get_api_names():
        return api
    else:
        raise ValueError(f'API {name} not compiled in')