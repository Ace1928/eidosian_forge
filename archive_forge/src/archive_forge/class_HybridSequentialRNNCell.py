from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class HybridSequentialRNNCell(HybridRecurrentCell):
    """Sequentially stacking multiple HybridRNN cells."""

    def __init__(self, prefix=None, params=None):
        super(HybridSequentialRNNCell, self).__init__(prefix=prefix, params=params)

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        return s.format(name=self.__class__.__name__, modstr='\n'.join(['({i}): {m}'.format(i=i, m=_indent(m.__repr__(), 2)) for i, m in self._children.items()]))

    def add(self, cell):
        """Appends a cell into the stack.

        Parameters
        ----------
        cell : RecurrentCell
            The cell to add.
        """
        self.register_child(cell)

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children.values(), batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, 'After applying modifier cells (e.g. ZoneoutCell) the base cell cannot be called directly. Call the modifier cell instead.'
        return _cells_begin_state(self._children.values(), **kwargs)

    def __call__(self, inputs, states):
        self._counter += 1
        next_states = []
        p = 0
        assert all((not isinstance(cell, BidirectionalCell) for cell in self._children.values()))
        for cell in self._children.values():
            n = len(cell.state_info())
            state = states[p:p + n]
            p += n
            inputs, state = cell(inputs, state)
            next_states.append(state)
        return (inputs, sum(next_states, []))

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None, valid_length=None):
        self.reset()
        inputs, _, F, batch_size = _format_sequence(length, inputs, layout, None)
        num_cells = len(self._children)
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)
        p = 0
        next_states = []
        for i, cell in enumerate(self._children.values()):
            n = len(cell.state_info())
            states = begin_state[p:p + n]
            p += n
            inputs, states = cell.unroll(length, inputs=inputs, begin_state=states, layout=layout, merge_outputs=None if i < num_cells - 1 else merge_outputs, valid_length=valid_length)
            next_states.extend(states)
        return (inputs, next_states)

    def __getitem__(self, i):
        return self._children[str(i)]

    def __len__(self):
        return len(self._children)

    def hybrid_forward(self, F, inputs, states):
        return self.__call__(inputs, states)