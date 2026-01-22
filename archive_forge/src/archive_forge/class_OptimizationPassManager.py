from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
from qiskit.transpiler.passes.optimization import (
from qiskit.transpiler.passes import Depth, Size, FixedPoint, MinimumPoint
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.circuit.library.standard_gates import (
class OptimizationPassManager(PassManagerStagePlugin):
    """Plugin class for optimization stage"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build pass manager for optimization stage."""
        translation_method = pass_manager_config.translation_method or 'translator'
        optimization = PassManager()
        if optimization_level != 0:
            plugin_manager = PassManagerStagePluginManager()
            _depth_check = [Depth(recurse=True), FixedPoint('depth')]
            _size_check = [Size(recurse=True), FixedPoint('size')]
            _minimum_point_check = [Depth(recurse=True), Size(recurse=True), MinimumPoint(['depth', 'size'], 'optimization_loop')]

            def _opt_control(property_set):
                return not property_set['depth_fixed_point'] or not property_set['size_fixed_point']
            translation = plugin_manager.get_passmanager_stage('translation', translation_method, pass_manager_config, optimization_level=optimization_level)
            if optimization_level == 1:
                _opt = [Optimize1qGatesDecomposition(basis=pass_manager_config.basis_gates, target=pass_manager_config.target), InverseCancellation([CXGate(), ECRGate(), CZGate(), CYGate(), XGate(), YGate(), ZGate(), HGate(), SwapGate(), (TGate(), TdgGate()), (SGate(), SdgGate()), (SXGate(), SXdgGate())])]
            elif optimization_level == 2:
                _opt = [Optimize1qGatesDecomposition(basis=pass_manager_config.basis_gates, target=pass_manager_config.target), CommutativeCancellation(basis_gates=pass_manager_config.basis_gates, target=pass_manager_config.target)]
            elif optimization_level == 3:
                _opt = [Collect2qBlocks(), ConsolidateBlocks(basis_gates=pass_manager_config.basis_gates, target=pass_manager_config.target, approximation_degree=pass_manager_config.approximation_degree), UnitarySynthesis(pass_manager_config.basis_gates, approximation_degree=pass_manager_config.approximation_degree, coupling_map=pass_manager_config.coupling_map, backend_props=pass_manager_config.backend_properties, method=pass_manager_config.unitary_synthesis_method, plugin_config=pass_manager_config.unitary_synthesis_plugin_config, target=pass_manager_config.target), Optimize1qGatesDecomposition(basis=pass_manager_config.basis_gates, target=pass_manager_config.target), CommutativeCancellation(target=pass_manager_config.target)]

                def _opt_control(property_set):
                    return not property_set['optimization_loop_minimum_point']
            else:
                raise TranspilerError(f'Invalid optimization_level: {optimization_level}')
            unroll = translation.to_flow_controller()

            def _unroll_condition(property_set):
                return not property_set['all_gates_in_basis']
            _unroll_if_out_of_basis = [GatesInBasis(pass_manager_config.basis_gates, target=pass_manager_config.target), ConditionalController(unroll, condition=_unroll_condition)]
            if optimization_level == 3:
                optimization.append(_minimum_point_check)
            else:
                optimization.append(_depth_check + _size_check)
            opt_loop = _opt + _unroll_if_out_of_basis + _minimum_point_check if optimization_level == 3 else _opt + _unroll_if_out_of_basis + _depth_check + _size_check
            optimization.append(DoWhileController(opt_loop, do_while=_opt_control))
            return optimization
        else:
            return None