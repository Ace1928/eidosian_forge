import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumScript
def _convert_measurement_to_numpy_data(m: qml.measurements.MeasurementProcess) -> qml.measurements.MeasurementProcess:
    if m.obs is None:
        if m.eigvals() is None or math.get_interface(m.eigvals()) == 'numpy':
            return m
        return type(m)(wires=m.wires, eigvals=math.unwrap(m.eigvals()))
    if math.get_interface(*m.obs.data) == 'numpy':
        return m
    new_obs = qml.ops.functions.bind_new_parameters(m.obs, math.unwrap(m.obs.data))
    return type(m)(obs=new_obs)