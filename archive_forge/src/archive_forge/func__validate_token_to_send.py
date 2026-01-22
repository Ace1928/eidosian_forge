from typing import Dict, Optional, Union
from .. import constants
from ._runtime import (
from ._token import get_token
from ._validators import validate_hf_hub_args
def _validate_token_to_send(token: Optional[str], is_write_action: bool) -> None:
    if is_write_action:
        if token is None:
            raise ValueError('Token is required (write-access action) but no token found. You need to provide a token or be logged in to Hugging Face with `huggingface-cli login` or `huggingface_hub.login`. See https://huggingface.co/settings/tokens.')
        if token.startswith('api_org'):
            raise ValueError('You must use your personal account token for write-access methods. To generate a write-access token, go to https://huggingface.co/settings/tokens')