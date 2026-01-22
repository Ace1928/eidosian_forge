from queue import Queue
from threading import Lock, Thread
from typing import Dict, Optional, Union
from urllib.parse import quote
from .. import constants, logging
from . import build_hf_headers, get_session, hf_raise_for_status
def _send_telemetry_in_thread(topic: str, *, library_name: Optional[str]=None, library_version: Optional[str]=None, user_agent: Union[Dict, str, None]=None) -> None:
    """Contains the actual data sending data to the Hub."""
    path = '/'.join((quote(part) for part in topic.split('/') if len(part) > 0))
    try:
        r = get_session().head(f'{constants.ENDPOINT}/api/telemetry/{path}', headers=build_hf_headers(token=False, library_name=library_name, library_version=library_version, user_agent=user_agent))
        hf_raise_for_status(r)
    except Exception as e:
        logger.debug(f'Error while sending telemetry: {e}')