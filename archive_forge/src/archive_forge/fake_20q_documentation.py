import os
from qiskit.providers.fake_provider import fake_qasm_backend
A fake backend with the following characteristics:

    * num_qubits: 20
    * coupling_map:

        .. code-block:: text

            00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
                  ↕         ↕
            05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
             ↕         ↕         ↕
            10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
                  ↕         ↕
            15 ↔ 16 ↔ 17 ↔ 18 ↔ 19

    * basis_gates: ``["id", "u1", "u2", "u3", "cx"]``
    