Dictionary containing commutation relations of unparameterizable standard gates. The dictionary key
is defined by a pair of gates and, optionally, their relative placement to each other, i.e.:

   ┌───┐┌───┐
q: ┤ X ├┤ Z ├    -> standard_gates_commutations["x", "z"] = False
   └───┘└───┘

or
          ┌───┐
q_0: ──■──┤ X ├
     ┌─┴─┐└─┬─┘  -> standard_gates_commutations["cx", "cy"][1, 0] = False
q_1: ┤ Y ├──■──
     └───┘

or

q_0: ──■───────
     ┌─┴─┐
q_1: ┤ X ├──■──  -> standard_gates_commutations['cx', 'cx'][None, 0] = False
     └───┘┌─┴─┐
q_2: ─────┤ X ├
          └───┘
Note that the commutation of a pair of gates is only stored in one specific order, i.e. the commutation
of a pair of cy and cx gates is only stored in standard_gates_commutations["cx", "cy"]. The order of the
gates is derived from the number of qubits in each gate and the value of the integer representation of
the gate name.

The values of standard_gates_commutations can be a Boolean value denoting whether the pair of gates
commute or another dictionary if the relative placement of the pair of gates has an impact on the
commutation. The relative placement is defined by the gate qubit arrangements as q2^{-1}[q1[i]] where
q1[i] is the ith qubit of the first gate and q2^{-1}[q] returns the qubit index of qubit q in the second
gate (possibly 'None'). In the second example, the zeroth qubit of cy overlaps with qubit 1 of cx. Non-
overlapping qubits are specified as 'None' in the dictionary key, e.g. example 3, there is no overlap
on the zeroth qubit of the first cx but the zeroth qubit of the second gate overlaps with qubit 1 of the
first cx.
