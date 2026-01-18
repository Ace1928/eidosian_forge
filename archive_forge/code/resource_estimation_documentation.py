from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.analysis.depth import Depth
from qiskit.transpiler.passes.analysis.width import Width
from qiskit.transpiler.passes.analysis.size import Size
from qiskit.transpiler.passes.analysis.count_ops import CountOps
from qiskit.transpiler.passes.analysis.num_tensor_factors import NumTensorFactors
from qiskit.transpiler.passes.analysis.num_qubits import NumQubits
Run the ResourceEstimation pass on `dag`.