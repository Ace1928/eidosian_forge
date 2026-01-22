from collections.abc import Iterable
import warnings
from typing import Sequence
def _ctrlo_circ(self, layer, wires, options=None):
    """Draw an open circle that indicates control on zero.

        Acceptable keys in options dictionary:
          * zorder
          * color
          * linewidth
        """
    new_options = _open_circ_options_process(options)
    circ_ctrlo = plt.Circle((layer, wires), radius=self._octrl_rad, **new_options)
    self._ax.add_patch(circ_ctrlo)