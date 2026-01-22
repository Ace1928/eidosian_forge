from typing import cast
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.api.v1 import params_pb2
def _to_zip_product(sweep: cirq.Sweep) -> cirq.Product:
    """Converts sweep to a product of zips of single sweeps, if possible."""
    if not isinstance(sweep, cirq.Product):
        sweep = cirq.Product(sweep)
    if not all((isinstance(f, cirq.Zip) for f in sweep.factors)):
        factors = [f if isinstance(f, cirq.Zip) else cirq.Zip(f) for f in sweep.factors]
        sweep = cirq.Product(*factors)
    for factor in sweep.factors:
        for term in cast(cirq.Zip, factor).sweeps:
            if not isinstance(term, sweeps.SingleSweep):
                raise ValueError(f'cannot convert to zip-product form: {sweep}')
    return sweep