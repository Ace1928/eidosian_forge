from __future__ import annotations
import typing
from abc import ABC, abstractmethod
from typing import Any
from qiskit.circuit.parameterexpression import ParameterExpression
Plot the interpolated envelope of pulse.

        Args:
            style: Stylesheet options. This can be dictionary or preset stylesheet classes. See
                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXStandard`,
                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXSimple`, and
                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXDebugging` for details of
                preset stylesheets.
            backend (Optional[BaseBackend]): Backend object to play the input pulse program.
                If provided, the plotter may use to make the visualization hardware aware.
            time_range: Set horizontal axis limit. Tuple ``(tmin, tmax)``.
            time_unit: The unit of specified time range either ``dt`` or ``ns``.
                The unit of ``ns`` is available only when ``backend`` object is provided.
            show_waveform_info: Show waveform annotations, i.e. name, of waveforms.
                Set ``True`` to show additional information about waveforms.
            plotter: Name of plotter API to generate an output image.
                One of following APIs should be specified::

                    mpl2d: Matplotlib API for 2D image generation.
                        Matplotlib API to generate 2D image. Charts are placed along y axis with
                        vertical offset. This API takes matplotlib.axes.Axes as `axis` input.

                `axis` and `style` kwargs may depend on the plotter.
            axis: Arbitrary object passed to the plotter. If this object is provided,
                the plotters use a given ``axis`` instead of internally initializing
                a figure object. This object format depends on the plotter.
                See plotter argument for details.

        Returns:
            Visualization output data.
            The returned data type depends on the ``plotter``.
            If matplotlib family is specified, this will be a ``matplotlib.pyplot.Figure`` data.
        