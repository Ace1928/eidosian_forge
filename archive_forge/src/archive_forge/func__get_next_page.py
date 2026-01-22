from typing import Dict, Iterable, Optional
import requests
from . import get_session, hf_raise_for_status, logging
def _get_next_page(response: requests.Response) -> Optional[str]:
    return response.links.get('next', {}).get('url')