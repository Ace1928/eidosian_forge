from typing import Any, Dict, Optional, Sequence
import dataclasses
from typing_extensions import Protocol
import cirq
@dataclasses.dataclass
class DeviceParameter(SupportsDeviceParameter):
    """Class for specifying device parameters.

    For instance, varying the length of pulses, timing, etc.
    This class is intended to be attached to a cirq.Points
    or cirq.Linspace sweep object as a metadata attribute.

    Args:
       path: path of the key to modify, with each sub-folder as a string
           entry in a list.
       idx: If this key is an array, which index to modify.
       value: value of the parameter to be set, if any.
       units: string value of the unit type of the value, if any.
          For instance, "GHz", "MHz", "ns", etc.
    """
    path: Sequence[str]
    idx: Optional[int] = None
    value: Optional[Any] = None
    units: Optional[str] = None

    def __repr__(self) -> str:
        return f'cirq_google.study.DeviceParameter(path={self.path!r}, idx={self.idx}, value={self.value!r}, units={self.units!r})'

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    @classmethod
    def _from_json_dict_(cls, path, idx, value, **kwargs):
        return DeviceParameter(path=path, idx=idx, value=value, units=kwargs['units'] if 'units' in kwargs else None)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['path', 'idx', 'value', 'units'])