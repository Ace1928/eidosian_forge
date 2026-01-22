import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
def _convert_to_pauli_words(self, observable):
    """Given an observable, obtain a list of coefficients and Pauli words, the
        sum of which is equal to the observable"""
    num_wires = self.bits.shape[1]
    obs_to_recipe_map = {'PauliX': 0, 'PauliY': 1, 'PauliZ': 2, 'Identity': -1}

    def pauli_list_to_word(obs):
        word = [-1] * num_wires
        for ob in obs:
            if ob.name not in obs_to_recipe_map:
                raise ValueError('Observable must be a linear combination of Pauli observables')
            word[self.wire_map.index(ob.wires[0])] = obs_to_recipe_map[ob.name]
        return word
    if isinstance(observable, (qml.X, qml.Y, qml.Z, qml.Identity)):
        word = pauli_list_to_word([observable])
        return [(1, word)]
    if isinstance(observable, qml.operation.Tensor):
        word = pauli_list_to_word(observable.obs)
        return [(1, word)]
    if isinstance(observable, qml.Hamiltonian):
        coeffs_and_words = []
        for coeff, op in zip(observable.data, observable.ops):
            coeffs_and_words.extend([(coeff * c, w) for c, w in self._convert_to_pauli_words(op)])
        return coeffs_and_words