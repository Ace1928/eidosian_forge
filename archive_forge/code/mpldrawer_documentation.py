from collections.abc import Iterable
import warnings
from typing import Sequence
Add classical communication double-lines for conditional operations

        Args:
            layer (int): the layer to draw vertical lines in, containing the target operation
            measured_layer (int): the layer where the mid-circuit measurements are
            wires (Union[int, Iterable[int]]): set of wires to control on
            wires_target (Union[int, Iterable[int]]): target wires. Used to determine where to
                terminate the vertical double-line

        Keyword Args:
            options=None (dict): Matplotlib keywords passed to ``plt.Line2D``

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=3, n_layers=4)

            drawer.cond(layer=1, measured_layer=0, wires=[0], wires_target=[1])

            options = {'color': "indigo", 'linewidth': 1.5}
            drawer.cond(layer=3, measured_layer=2, wires=(1,), wires_target=(2,), options=options)

        .. figure:: ../../_static/drawer/cond.png
            :align: center
            :width: 60%
            :target: javascript:void(0);
        