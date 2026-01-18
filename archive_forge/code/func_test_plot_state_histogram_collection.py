import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pytest
import cirq
from cirq.devices import GridQubit
from cirq.vis import state_histogram
@pytest.mark.usefixtures('closefigures')
def test_plot_state_histogram_collection():
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(cirq.X.on_each(*qubits[1:]), cirq.measure(*qubits))
    r = cirq.sample(c, repetitions=5)
    _, (ax1, ax2) = plt.subplots(1, 2)
    state_histogram.plot_state_histogram(r.histogram(key='q(0),q(1),q(2),q(3)'), ax1)
    expected_values = [5]
    tick_label = ['7']
    state_histogram.plot_state_histogram(expected_values, ax2, tick_label=tick_label, xlabel=None)
    for r1, r2 in zip(ax1.get_children(), ax2.get_children()):
        if isinstance(r1, mpl.patches.Rectangle) and isinstance(r2, mpl.patches.Rectangle):
            assert str(r1) == str(r2)