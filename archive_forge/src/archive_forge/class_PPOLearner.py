from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from ray.rllib.core.learner.learner import Learner, LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.schedules.scheduler import Scheduler
class PPOLearner(Learner):

    @override(Learner)
    def build(self) -> None:
        super().build()
        self.entropy_coeff_schedulers_per_module: Dict[ModuleID, Scheduler] = LambdaDefaultDict(lambda module_id: Scheduler(fixed_value_or_schedule=self.hps.get_hps_for_module(module_id).entropy_coeff, framework=self.framework, device=self._device))
        self.curr_kl_coeffs_per_module: Dict[ModuleID, Scheduler] = LambdaDefaultDict(lambda module_id: self._get_tensor_variable(self.hps.get_hps_for_module(module_id).kl_coeff))

    @override(Learner)
    def remove_module(self, module_id: str):
        super().remove_module(module_id)
        self.curr_kl_coeffs_per_module.pop(module_id)
        self.entropy_coeff_schedulers_per_module.pop(module_id)

    @override(Learner)
    def additional_update_for_module(self, *, module_id: ModuleID, hps: PPOLearnerHyperparameters, timestep: int, sampled_kl_values: dict) -> Dict[str, Any]:
        results = super().additional_update_for_module(module_id=module_id, hps=hps, timestep=timestep, sampled_kl_values=sampled_kl_values)
        new_entropy_coeff = self.entropy_coeff_schedulers_per_module[module_id].update(timestep=timestep)
        results.update({LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY: new_entropy_coeff})
        return results