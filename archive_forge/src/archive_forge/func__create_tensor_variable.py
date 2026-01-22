from typing import Optional
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.schedules.piecewise_schedule import PiecewiseSchedule
from ray.rllib.utils.typing import LearningRateOrSchedule, TensorType
def _create_tensor_variable(self, initial_value: float) -> TensorType:
    """Creates a framework-specific tensor variable to be scheduled.

        Args:
            initial_value: The initial (float) value for the variable to hold.

        Returns:
            The created framework-specific tensor variable.
        """
    if self.framework == 'torch':
        return torch.tensor(initial_value, requires_grad=False, dtype=torch.float32, device=self.device)
    else:
        return tf.Variable(initial_value, trainable=False, dtype=tf.float32)