from __future__ import annotations
from typing import (
from pandas.core.interchange.dataframe_protocol import (
def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
    """
        Device type and device ID for where the data in the buffer resides.
        """
    return (DlpackDeviceType.CPU, None)