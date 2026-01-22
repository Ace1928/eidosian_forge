from collections.abc import Iterable
import warnings
from typing import Sequence
def _target_x(self, layer, wires, options=None):
    """Draws the circle used to represent a CNOT's target

        Args:
            layer (int): layer to draw on
            wires (int): wire to draw on

        Keyword Args:
            options=None (dict): Matplotlib keywords. The only supported keys are ``'color'``, ``'linewidth'``,
                and ``'zorder'``.
        """
    if options is None:
        options = {}
    new_options = _open_circ_options_process(options)
    options['zorder'] = new_options['zorder'] + 1
    target_circ = plt.Circle((layer, wires), radius=self._circ_rad, **new_options)
    target_v = plt.Line2D((layer, layer), (wires - self._circ_rad, wires + self._circ_rad), **options)
    target_h = plt.Line2D((layer - self._circ_rad, layer + self._circ_rad), (wires, wires), **options)
    self._ax.add_patch(target_circ)
    self._ax.add_line(target_v)
    self._ax.add_line(target_h)