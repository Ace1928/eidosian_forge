from typing import Dict, Any, Mapping
from qiskit.visualization.pulse_v2 import generators, layouts
class QiskitPulseStyle(dict):
    """Stylesheet for pulse drawer."""

    def __init__(self):
        super().__init__()
        self.stylesheet = None
        self.update(default_style())

    def update(self, __m: Mapping[str, Any], **kwargs) -> None:
        super().update(__m, **kwargs)
        for key, value in __m.items():
            self.__setitem__(key, value)
        self.stylesheet = __m.__class__.__name__

    @property
    def formatter(self):
        """Return formatter field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split('.')
            if sub_keys[0] == 'formatter':
                sub_dict['.'.join(sub_keys[1:])] = value
        return sub_dict

    @property
    def generator(self):
        """Return generator field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split('.')
            if sub_keys[0] == 'generator':
                sub_dict['.'.join(sub_keys[1:])] = value
        return sub_dict

    @property
    def layout(self):
        """Return layout field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split('.')
            if sub_keys[0] == 'layout':
                sub_dict['.'.join(sub_keys[1:])] = value
        return sub_dict