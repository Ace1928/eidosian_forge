from __future__ import annotations
from typing import TYPE_CHECKING
from pymongo.errors import ConfigurationError
from pymongo.server_type import SERVER_TYPE
def _with_primary(max_staleness: int, selection: Selection) -> Selection:
    """Apply max_staleness, in seconds, to a Selection with a known primary."""
    primary = selection.primary
    assert primary
    sds = []
    for s in selection.server_descriptions:
        if s.server_type == SERVER_TYPE.RSSecondary:
            assert s.last_write_date and primary.last_write_date
            staleness = s.last_update_time - s.last_write_date - (primary.last_update_time - primary.last_write_date) + selection.heartbeat_frequency
            if staleness <= max_staleness:
                sds.append(s)
        else:
            sds.append(s)
    return selection.with_server_descriptions(sds)