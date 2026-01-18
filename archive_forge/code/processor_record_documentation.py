import abc
import dataclasses
from typing import Any, Dict
import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr
from cirq_google.engine.virtual_engine_factory import (
Return a `cg.GridDevice` for the specified processor_id.

        Only 'rainbow' and 'weber' are recognized processor_ids and the device information
        may not be up-to-date, as it is completely local.
        