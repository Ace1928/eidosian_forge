import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
class T2DecayResult:
    """Results from a T2 decay experiment.

    This object is a container for the measurement results in each basis
    for each amount of delay.  These can be used to calculate Pauli
    expectation values, length of the Bloch vector, and various fittings of
    the data to calculate estimated T2 phase decay times.
    """

    def __init__(self, x_basis_data: pd.DataFrame, y_basis_data: pd.DataFrame):
        """Inits T2DecayResult.

        Args:
            x_basis_data: Data frame in x basis with three columns: delay_ns,
                false_count, and true_count.
            y_basis_data: Data frame in y basis with three columns: delay_ns,
                false_count,  and true_count.

        Raises:
            ValueError: If the supplied data does not have the proper columns.
        """
        x_cols = list(x_basis_data.columns)
        y_cols = list(y_basis_data.columns)
        if any((col not in x_cols for col in _T2_COLUMNS)):
            raise ValueError(f'x_basis_data must have columns {_T2_COLUMNS} but had {list(x_basis_data.columns)}')
        if any((col not in y_cols for col in _T2_COLUMNS)):
            raise ValueError(f'y_basis_data must have columns {_T2_COLUMNS} but had {list(y_basis_data.columns)}')
        self._x_basis_data = x_basis_data
        self._y_basis_data = y_basis_data
        self._expectation_pauli_x = self._expectation(x_basis_data)
        self._expectation_pauli_y = self._expectation(y_basis_data)

    def _expectation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the expected value of the Pauli operator.

        Assuming that the data is measured in the Pauli basis of the operator,
        then the expectation of the Pauli operator would be +1 if the
        measurement is all ones and -1 if the measurement is all zeros.

        Args:
            data: measurement data to compute the expecation for.

        Returns:
            Data frame with columns 'delay_ns', 'num_pulses' and 'value'
            The num_pulses column will only exist if multiple pulses
            were requestd in the T2 experiment.
        """
        delay = data['delay_ns']
        ones = data[1]
        zeros = data[0]
        pauli_expectation = 2 * (ones / (ones + zeros)) - 1.0
        if 'num_pulses' in data.columns:
            return pd.DataFrame({'delay_ns': delay, 'num_pulses': data['num_pulses'], 'value': pauli_expectation})
        return pd.DataFrame({'delay_ns': delay, 'value': pauli_expectation})

    @property
    def expectation_pauli_x(self) -> pd.DataFrame:
        """A data frame with delay_ns, value columns.

        This value contains the expectation of the Pauli X operator as
        estimated by measurement outcomes.
        """
        return self._expectation_pauli_x

    @property
    def expectation_pauli_y(self) -> pd.DataFrame:
        """A data frame with delay_ns, value columns.

        This value contains the expectation of the Pauli X operator as
        estimated by measurement outcomes.
        """
        return self._expectation_pauli_y

    def plot_expectations(self, ax: Optional[plt.Axes]=None, **plot_kwargs: Any) -> plt.Axes:
        """Plots the expectation values of Pauli operators versus delay time.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        assert ax is not None
        ax.set_ylim(ymin=-2, ymax=2)
        ax.plot(self._expectation_pauli_x['delay_ns'], self._expectation_pauli_x['value'], 'bo-', label='<X>', **plot_kwargs)
        ax.plot(self._expectation_pauli_y['delay_ns'], self._expectation_pauli_y['value'], 'go-', label='<Y>', **plot_kwargs)
        ax.set_xlabel('Delay between initialization and measurement (nanoseconds)')
        ax.set_ylabel('Pauli Operator Expectation')
        ax.set_title('T2 Decay Pauli Expectations')
        ax.legend()
        if show_plot:
            fig.show()
        return ax

    def plot_bloch_vector(self, ax: Optional[plt.Axes]=None, **plot_kwargs: Any) -> plt.Axes:
        """Plots the estimated length of the Bloch vector versus time.

        This plot estimates the Bloch Vector by squaring the Pauli expectation
        value of X and adding it to the square of the Pauli expectation value of
        Y.  This essentially projects the state into the XY plane.

        Note that Z expectation is not considered, since T1 related amplitude
        damping will generally push this value towards |0>
        (expectation <Z> = -1) which will significantly distort the T2 numbers.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        assert ax is not None
        ax.set_ylim(ymin=0, ymax=1)
        bloch_vector = self._expectation_pauli_x ** 2 + self._expectation_pauli_y ** 2
        ax.plot(self._expectation_pauli_x['delay_ns'], bloch_vector['value'], 'r+-', **plot_kwargs)
        ax.set_xlabel('Delay between initialization and measurement (nanoseconds)')
        ax.set_ylabel('Bloch Vector X-Y Projection Squared')
        ax.set_title('T2 Decay Experiment Data')
        if show_plot:
            fig.show()
        return ax

    def __str__(self):
        return f'T2DecayResult with data:\n<X>\n{self._x_basis_data}\n<Y>\n{self._y_basis_data}'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._expectation_pauli_x.equals(other._expectation_pauli_x) and self._expectation_pauli_y.equals(other._expectation_pauli_y)

    def __repr__(self):
        return f'cirq.experiments.T2DecayResult(x_basis_data={proper_repr(self._x_basis_data)}, y_basis_data={proper_repr(self._y_basis_data)})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            p.text('T2DecayResult(...)')
        else:
            p.text(str(self))