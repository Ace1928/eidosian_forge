import abc
from typing import List, Optional, Dict
import stevedore
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager_config import PassManagerConfig
def _build_pm(self, stage_obj: stevedore.ExtensionManager, stage_name: str, plugin_name: str, pm_config: PassManagerConfig, optimization_level: Optional[int]=None):
    if plugin_name not in stage_obj:
        raise TranspilerError(f'Invalid plugin name {plugin_name} for stage {stage_name}')
    plugin_obj = stage_obj[plugin_name]
    return plugin_obj.obj.pass_manager(pm_config, optimization_level)