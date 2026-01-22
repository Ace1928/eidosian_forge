import os
from qiskit.providers.fake_provider import fake_qasm_backend
A fake backend with the following characteristics:

    * num_qubits: 5
    * coupling_map:

        .. code-block:: text

                1
              / |
            0 - 2 - 3
                | /
                4

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    