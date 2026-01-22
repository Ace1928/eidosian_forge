from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
def _server_accepts_client_ids() -> bool:
    from wandb.util import parse_version
    if util._is_offline():
        return False
    max_cli_version = util._get_max_cli_version()
    if max_cli_version is None:
        return False
    accepts_client_ids: bool = parse_version('0.11.0') <= parse_version(max_cli_version)
    return accepts_client_ids