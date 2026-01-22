from __future__ import annotations
from typing import Any, MutableMapping, Optional
def _add_to_command(cmd: MutableMapping[str, Any], server_api: Optional[ServerApi]) -> None:
    """Internal helper which adds API versioning options to a command.

    :Parameters:
      - `cmd`: The command.
      - `server_api` (optional): A :class:`ServerApi` or ``None``.
    """
    if not server_api:
        return
    cmd['apiVersion'] = server_api.version
    if server_api.strict is not None:
        cmd['apiStrict'] = server_api.strict
    if server_api.deprecation_errors is not None:
        cmd['apiDeprecationErrors'] = server_api.deprecation_errors