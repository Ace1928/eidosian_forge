from gymnasium.spaces import Space
import numpy as np
from typing import Union, Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import (
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import Schedule
from ray.rllib.utils.schedules.piecewise_schedule import PiecewiseSchedule
from ray.rllib.utils.tf_utils import zero_logps_from_actions
@PublicAPI
class GaussianNoise(Exploration):
    """An exploration that adds white noise to continuous actions.

    If explore=True, returns actions plus scale (annealed over time) x
    Gaussian noise. Also, some completely random period is possible at the
    beginning.

    If explore=False, returns the deterministic action.
    """

    def __init__(self, action_space: Space, *, framework: str, model: ModelV2, random_timesteps: int=1000, stddev: float=0.1, initial_scale: float=1.0, final_scale: float=0.02, scale_timesteps: int=10000, scale_schedule: Optional[Schedule]=None, **kwargs):
        """Initializes a GaussianNoise instance.

        Args:
            random_timesteps: The number of timesteps for which to act
                completely randomly. Only after this number of timesteps, the
                `self.scale` annealing process will start (see below).
            stddev: The stddev (sigma) to use for the
                Gaussian noise to be added to the actions.
            initial_scale: The initial scaling weight to multiply
                the noise with.
            final_scale: The final scaling weight to multiply
                the noise with.
            scale_timesteps: The timesteps over which to linearly anneal
                the scaling factor (after(!) having used random actions for
                `random_timesteps` steps).
            scale_schedule: An optional Schedule object
                to use (instead of constructing one from the given parameters).
        """
        assert framework is not None
        super().__init__(action_space, model=model, framework=framework, **kwargs)
        self.random_timesteps = random_timesteps
        self.random_exploration = Random(action_space, model=self.model, framework=self.framework, **kwargs)
        self.stddev = stddev
        self.scale_schedule = scale_schedule or PiecewiseSchedule(endpoints=[(random_timesteps, initial_scale), (random_timesteps + scale_timesteps, final_scale)], outside_value=final_scale, framework=self.framework)
        self.last_timestep = get_variable(np.array(0, np.int64), framework=self.framework, tf_name='timestep', dtype=np.int64)
        if self.framework == 'tf':
            self._tf_state_op = self.get_state()

    @override(Exploration)
    def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Union[int, TensorType], explore: bool=True):
        if self.framework == 'torch':
            return self._get_torch_exploration_action(action_distribution, explore, timestep)
        else:
            return self._get_tf_exploration_action_op(action_distribution, explore, timestep)

    def _get_tf_exploration_action_op(self, action_dist: ActionDistribution, explore: bool, timestep: Union[int, TensorType]):
        ts = timestep if timestep is not None else self.last_timestep
        deterministic_actions = action_dist.deterministic_sample()
        gaussian_sample = self.scale_schedule(ts) * tf.random.normal(tf.shape(deterministic_actions), stddev=self.stddev)
        random_actions, _ = self.random_exploration.get_tf_exploration_action_op(action_dist, explore)
        stochastic_actions = tf.cond(pred=tf.convert_to_tensor(ts < self.random_timesteps), true_fn=lambda: random_actions, false_fn=lambda: tf.clip_by_value(deterministic_actions + gaussian_sample, self.action_space.low * tf.ones_like(deterministic_actions), self.action_space.high * tf.ones_like(deterministic_actions)))
        action = tf.cond(pred=tf.constant(explore, dtype=tf.bool) if isinstance(explore, bool) else explore, true_fn=lambda: stochastic_actions, false_fn=lambda: deterministic_actions)
        logp = zero_logps_from_actions(deterministic_actions)
        if self.framework == 'tf2':
            if timestep is None:
                self.last_timestep.assign_add(1)
            else:
                self.last_timestep.assign(tf.cast(timestep, tf.int64))
            return (action, logp)
        else:
            assign_op = tf1.assign_add(self.last_timestep, 1) if timestep is None else tf1.assign(self.last_timestep, timestep)
            with tf1.control_dependencies([assign_op]):
                return (action, logp)

    def _get_torch_exploration_action(self, action_dist: ActionDistribution, explore: bool, timestep: Union[int, TensorType]):
        self.last_timestep = timestep if timestep is not None else self.last_timestep + 1
        if explore:
            if self.last_timestep < self.random_timesteps:
                action, _ = self.random_exploration.get_torch_exploration_action(action_dist, explore=True)
            else:
                det_actions = action_dist.deterministic_sample()
                scale = self.scale_schedule(self.last_timestep)
                gaussian_sample = scale * torch.normal(mean=torch.zeros(det_actions.size()), std=self.stddev).to(self.device)
                action = torch.min(torch.max(det_actions + gaussian_sample, torch.tensor(self.action_space.low, dtype=torch.float32, device=self.device)), torch.tensor(self.action_space.high, dtype=torch.float32, device=self.device))
        else:
            action = action_dist.deterministic_sample()
        logp = torch.zeros((action.size()[0],), dtype=torch.float32, device=self.device)
        return (action, logp)

    @override(Exploration)
    def get_state(self, sess: Optional['tf.Session']=None):
        """Returns the current scale value.

        Returns:
            Union[float,tf.Tensor[float]]: The current scale value.
        """
        if sess:
            return sess.run(self._tf_state_op)
        scale = self.scale_schedule(self.last_timestep)
        return {'cur_scale': convert_to_numpy(scale) if self.framework != 'tf' else scale, 'last_timestep': convert_to_numpy(self.last_timestep) if self.framework != 'tf' else self.last_timestep}

    @override(Exploration)
    def set_state(self, state: dict, sess: Optional['tf.Session']=None) -> None:
        if self.framework == 'tf':
            self.last_timestep.load(state['last_timestep'], session=sess)
        elif isinstance(self.last_timestep, int):
            self.last_timestep = state['last_timestep']
        else:
            self.last_timestep.assign(state['last_timestep'])