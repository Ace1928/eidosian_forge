from typing import Iterator
import networkx as nx
import cirq
For low weight operators on low-degree circuits, we can simplify
    the circuit representation of an expectation value.

    In particular, this should be used on `circuit_for_expectation_value`
    circuits. It will merge single- and two-qubit gates from the "forwards"
    and "backwards" parts of the circuit outside of the operator's lightcone.

    This might be too slow in practice and you can just use quimb to simplify
    things for you.
    