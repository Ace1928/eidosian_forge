from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def _plot_wires(self):
    """Plot the wires of the circuit diagram."""
    xstart = self._gate_grid[0]
    xstop = self._gate_grid[-1]
    xdata = (xstart - self.scale, xstop + self.scale)
    for i in range(self.nqubits):
        ydata = (self._wire_grid[i], self._wire_grid[i])
        line = Line2D(xdata, ydata, color='k', lw=self.linewidth)
        self._axes.add_line(line)
        if self.labels:
            init_label_buffer = 0
            if self.inits.get(self.labels[i]):
                init_label_buffer = 0.25
            self._axes.text(xdata[0] - self.label_buffer - init_label_buffer, ydata[0], render_label(self.labels[i], self.inits), size=self.fontsize, color='k', ha='center', va='center')
    self._plot_measured_wires()