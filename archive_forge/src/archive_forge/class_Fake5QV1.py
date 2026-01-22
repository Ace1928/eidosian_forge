import os
from qiskit.providers.fake_provider import fake_qasm_backend
class Fake5QV1(fake_qasm_backend.FakeQasmBackend):
    """A fake backend with the following characteristics:

    * num_qubits: 5
    * coupling_map:

        .. code-block:: text

                1
              / |
            0 - 2 - 3
                | /
                4

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    """
    dirname = os.path.dirname(__file__)
    conf_filename = 'conf_yorktown.json'
    props_filename = 'props_yorktown.json'
    backend_name = 'fake_5q_v1'