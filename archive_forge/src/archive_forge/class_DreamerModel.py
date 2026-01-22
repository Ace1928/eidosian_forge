import re
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.dreamerv3.tf.models.disagree_networks import DisagreeNetworks
from ray.rllib.algorithms.dreamerv3.tf.models.actor_network import ActorNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.critic_network import CriticNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.world_model import WorldModel
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import inverse_symlog
class DreamerModel(tf.keras.Model):
    """The main tf-keras model containing all necessary components for DreamerV3.

    Includes:
    - The world model with encoder, decoder, sequence-model (RSSM), dynamics
    (generates prior z-state), and "posterior" model (generates posterior z-state).
    Predicts env dynamics and produces dreamed trajectories for actor- and critic
    learning.
    - The actor network (policy).
    - The critic network for value function prediction.
    """

    def __init__(self, *, model_size: str='XS', action_space: gym.Space, world_model: WorldModel, actor: ActorNetwork, critic: CriticNetwork, horizon: int, gamma: float, use_curiosity: bool=False, intrinsic_rewards_scale: float=0.1):
        """Initializes a DreamerModel instance.

        Args:
             model_size: The "Model Size" used according to [1] Appendinx B.
                Use None for manually setting the different network sizes.
             action_space: The action space the our environment used.
             world_model: The WorldModel component.
             actor: The ActorNetwork component.
             critic: The CriticNetwork component.
             horizon: The dream horizon to use when creating dreamed trajectories.
        """
        super().__init__(name='dreamer_model')
        self.model_size = model_size
        self.action_space = action_space
        self.use_curiosity = use_curiosity
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.horizon = horizon
        self.gamma = gamma
        self._comp_dtype = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.disagree_nets = None
        if self.use_curiosity:
            self.disagree_nets = DisagreeNetworks(num_networks=8, model_size=self.model_size, intrinsic_rewards_scale=intrinsic_rewards_scale)
        self.dream_trajectory = tf.function(input_signature=[{'h': tf.TensorSpec(shape=[None, get_gru_units(self.model_size)], dtype=self._comp_dtype), 'z': tf.TensorSpec(shape=[None, get_num_z_categoricals(self.model_size), get_num_z_classes(self.model_size)], dtype=self._comp_dtype)}, tf.TensorSpec(shape=[None], dtype=tf.bool)])(self.dream_trajectory)

    def call(self, inputs, observations, actions, is_first, start_is_terminated_BxT):
        """Main call method for building this model in order to generate its variables.

        Note: This method should NOT be used by users directly. It's purpose is only to
        perform all forward passes necessary to define all variables of the DreamerV3.
        """
        results = self.world_model.forward_train(observations, actions, is_first)
        _, distr_params = self.actor(h=results['h_states_BxT'], z=results['z_posterior_states_BxT'])
        values, _ = self.critic(h=results['h_states_BxT'], z=results['z_posterior_states_BxT'], use_ema=tf.convert_to_tensor(False))
        dream_data = self.dream_trajectory(start_states={'h': results['h_states_BxT'], 'z': results['z_posterior_states_BxT']}, start_is_terminated=start_is_terminated_BxT)
        return {'world_model_fwd': results, 'dream_data': dream_data, 'actions': actions, 'values': values}

    @tf.function
    def forward_inference(self, observations, previous_states, is_first, training=None):
        """Performs a (non-exploring) action computation step given obs and states.

        Note that all input data should not have a time rank (only a batch dimension).

        Args:
            observations: The current environment observation with shape (B, ...).
            previous_states: Dict with keys `a`, `h`, and `z` used as input to the RSSM
                to produce the next h-state, from which then to compute the action
                using the actor network. All values in the dict should have shape
                (B, ...) (no time rank).
            is_first: Batch of is_first flags. These should be True if a new episode
                has been started at the current timestep (meaning `observations` is the
                reset observation from the environment).
        """
        states = self.world_model.forward_inference(observations=observations, previous_states=previous_states, is_first=is_first)
        _, distr_params = self.actor(h=states['h'], z=states['z'])
        distr = self.actor.get_action_dist_object(distr_params)
        actions = distr.mode()
        return (actions, {'h': states['h'], 'z': states['z'], 'a': actions})

    @tf.function
    def forward_exploration(self, observations, previous_states, is_first, training=None):
        """Performs an exploratory action computation step given obs and states.

        Note that all input data should not have a time rank (only a batch dimension).

        Args:
            observations: The current environment observation with shape (B, ...).
            previous_states: Dict with keys `a`, `h`, and `z` used as input to the RSSM
                to produce the next h-state, from which then to compute the action
                using the actor network. All values in the dict should have shape
                (B, ...) (no time rank).
            is_first: Batch of is_first flags. These should be True if a new episode
                has been started at the current timestep (meaning `observations` is the
                reset observation from the environment).
        """
        states = self.world_model.forward_inference(observations=observations, previous_states=previous_states, is_first=is_first)
        actions, _ = self.actor(h=states['h'], z=states['z'])
        return (actions, {'h': states['h'], 'z': states['z'], 'a': actions})

    def forward_train(self, observations, actions, is_first):
        """Performs a training forward pass given observations and actions.

        Note that all input data must have a time rank (batch-major: [B, T, ...]).

        Args:
            observations: The environment observations with shape (B, T, ...). Thus,
                the batch has B rows of T timesteps each. Note that it's ok to have
                episode boundaries (is_first=True) within a batch row. DreamerV3 will
                simply insert an initial state before these locations and continue the
                sequence modelling (with the RSSM). Hence, there will be no zero
                padding.
            actions: The actions actually taken in the environment with shape
                (B, T, ...). See `observations` docstring for details on how B and T are
                handled.
            is_first: Batch of is_first flags. These should be True:
                - if a new episode has been started at the current timestep (meaning
                `observations` is the reset observation from the environment).
                - in each batch row at T=0 (first timestep of each of the B batch
                rows), regardless of whether the actual env had an episode boundary
                there or not.
        """
        return self.world_model.forward_train(observations=observations, actions=actions, is_first=is_first)

    @tf.function
    def get_initial_state(self):
        """Returns the (current) initial state of the dreamer model (a, h-, z-states).

        An initial state is generated using the previous action, the tanh of the
        (learned) h-state variable and the dynamics predictor (or "prior net") to
        compute z^0 from h0. In this last step, it is important that we do NOT sample
        the z^-state (as we would usually do during dreaming), but rather take the mode
        (argmax, then one-hot again).
        """
        states = self.world_model.get_initial_state()
        action_dim = self.action_space.n if isinstance(self.action_space, gym.spaces.Discrete) else np.prod(self.action_space.shape)
        states['a'] = tf.zeros((1, action_dim), dtype=tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32)
        return states

    def dream_trajectory(self, start_states, start_is_terminated):
        """Dreams trajectories of length H from batch of h- and z-states.

        Note that incoming data will have the shapes (BxT, ...), where the original
        batch- and time-dimensions are already folded together. Beginning from this
        new batch dim (BxT), we will unroll `timesteps_H` timesteps in a time-major
        fashion, such that the dreamed data will have shape (H, BxT, ...).

        Args:
            start_states: Dict of `h` and `z` states in the shape of (B, ...) and
                (B, num_categoricals, num_classes), respectively, as
                computed by a train forward pass. From each individual h-/z-state pair
                in the given batch, we will branch off a dreamed trajectory of len
                `timesteps_H`.
            start_is_terminated: Float flags of shape (B,) indicating whether the
                first timesteps of each batch row is already a terminated timestep
                (given by the actual environment).
        """
        a_dreamed_t0_to_H = []
        a_dreamed_dist_params_t0_to_H = []
        h = start_states['h']
        z = start_states['z']
        h_states_t0_to_H = [h]
        z_states_prior_t0_to_H = [z]
        a, a_dist_params = self.actor(h=tf.stop_gradient(h), z=tf.stop_gradient(z))
        a_dreamed_t0_to_H.append(a)
        a_dreamed_dist_params_t0_to_H.append(a_dist_params)
        for i in range(self.horizon):
            h = self.world_model.sequence_model(a=a, h=h, z=z)
            h_states_t0_to_H.append(h)
            z, _ = self.world_model.dynamics_predictor(h=h)
            z_states_prior_t0_to_H.append(z)
            a, a_dist_params = self.actor(h=tf.stop_gradient(h), z=tf.stop_gradient(z))
            a_dreamed_t0_to_H.append(a)
            a_dreamed_dist_params_t0_to_H.append(a_dist_params)
        h_states_H_B = tf.stack(h_states_t0_to_H, axis=0)
        h_states_HxB = tf.reshape(h_states_H_B, [-1] + h_states_H_B.shape.as_list()[2:])
        z_states_prior_H_B = tf.stack(z_states_prior_t0_to_H, axis=0)
        z_states_prior_HxB = tf.reshape(z_states_prior_H_B, [-1] + z_states_prior_H_B.shape.as_list()[2:])
        a_dreamed_H_B = tf.stack(a_dreamed_t0_to_H, axis=0)
        a_dreamed_dist_params_H_B = tf.stack(a_dreamed_dist_params_t0_to_H, axis=0)
        r_dreamed_HxB, _ = self.world_model.reward_predictor(h=h_states_HxB, z=z_states_prior_HxB)
        r_dreamed_H_B = tf.reshape(inverse_symlog(r_dreamed_HxB), shape=[self.horizon + 1, -1])
        if self.use_curiosity:
            results_HxB = self.disagree_nets.compute_intrinsic_rewards(h=h_states_HxB, z=z_states_prior_HxB, a=tf.reshape(a_dreamed_H_B, [-1] + a_dreamed_H_B.shape.as_list()[2:]))
            r_intrinsic_H_B = tf.reshape(results_HxB['rewards_intrinsic'], shape=[self.horizon + 1, -1])[1:]
            curiosity_forward_train_outs = results_HxB['forward_train_outs']
            del results_HxB
        c_dreamed_HxB, _ = self.world_model.continue_predictor(h=h_states_HxB, z=z_states_prior_HxB)
        c_dreamed_H_B = tf.reshape(c_dreamed_HxB, [self.horizon + 1, -1])
        c_dreamed_H_B = tf.concat([1.0 - tf.expand_dims(tf.cast(start_is_terminated, tf.float32), 0), c_dreamed_H_B[1:]], axis=0)
        dream_loss_weights_H_B = tf.math.cumprod(self.gamma * c_dreamed_H_B, axis=0) / self.gamma
        v, v_symlog_dreamed_logits_HxB = self.critic(h=h_states_HxB, z=z_states_prior_HxB, use_ema=False)
        v_dreamed_HxB = inverse_symlog(v)
        v_dreamed_H_B = tf.reshape(v_dreamed_HxB, shape=[self.horizon + 1, -1])
        v_symlog_dreamed_ema_HxB, _ = self.critic(h=h_states_HxB, z=z_states_prior_HxB, use_ema=True)
        v_symlog_dreamed_ema_H_B = tf.reshape(v_symlog_dreamed_ema_HxB, shape=[self.horizon + 1, -1])
        ret = {'h_states_t0_to_H_BxT': h_states_H_B, 'z_states_prior_t0_to_H_BxT': z_states_prior_H_B, 'rewards_dreamed_t0_to_H_BxT': r_dreamed_H_B, 'continues_dreamed_t0_to_H_BxT': c_dreamed_H_B, 'actions_dreamed_t0_to_H_BxT': a_dreamed_H_B, 'actions_dreamed_dist_params_t0_to_H_BxT': a_dreamed_dist_params_H_B, 'values_dreamed_t0_to_H_BxT': v_dreamed_H_B, 'values_symlog_dreamed_logits_t0_to_HxBxT': v_symlog_dreamed_logits_HxB, 'v_symlog_dreamed_ema_t0_to_H_BxT': v_symlog_dreamed_ema_H_B, 'dream_loss_weights_t0_to_H_BxT': dream_loss_weights_H_B}
        if self.use_curiosity:
            ret['rewards_intrinsic_t1_to_H_B'] = r_intrinsic_H_B
            ret.update(curiosity_forward_train_outs)
        if isinstance(self.action_space, gym.spaces.Discrete):
            ret['actions_ints_dreamed_t0_to_H_B'] = tf.argmax(a_dreamed_H_B, axis=-1)
        return ret

    def dream_trajectory_with_burn_in(self, *, start_states, timesteps_burn_in: int, timesteps_H: int, observations, actions, use_sampled_actions_in_dream: bool=False, use_random_actions_in_dream: bool=False):
        """Dreams trajectory from N initial observations and initial states.

        Note: This is only used for reporting and debugging, not for actual world-model
        or policy training.

        Args:
            start_states: The batch of start states (dicts with `a`, `h`, and `z` keys)
                to begin dreaming with. These are used to compute the first h-state
                using the sequence model.
            timesteps_burn_in: For how many timesteps should be use the posterior
                z-states (computed by the posterior net and actual observations from
                the env)?
            timesteps_H: For how many timesteps should we dream using the prior
                z-states (computed by the dynamics (prior) net and h-states only)?
                Note that the total length of the returned trajectories will
                be `timesteps_burn_in` + `timesteps_H`.
            observations: The batch (B, T, ...) of observations (to be used only during
                burn-in over `timesteps_burn_in` timesteps).
            actions: The batch (B, T, ...) of actions to use during a) burn-in over the
                first `timesteps_burn_in` timesteps and - possibly - b) during
                actual dreaming, iff use_sampled_actions_in_dream=True.
                If applicable, actions must already be one-hot'd.
            use_sampled_actions_in_dream: If True, instead of using our actor network
                to compute fresh actions, we will use the one provided via the `actions`
                argument. Note that in the latter case, the `actions` time dimension
                must be at least `timesteps_burn_in` + `timesteps_H` long.
            use_random_actions_in_dream: Whether to use randomly sampled actions in the
                dream. Note that this does not apply to the burn-in phase, during which
                we will always use the actions given in the `actions` argument.
        """
        assert not (use_sampled_actions_in_dream and use_random_actions_in_dream)
        B = observations.shape[0]
        states = start_states
        for i in range(timesteps_burn_in):
            states = self.world_model.forward_inference(observations=observations[:, i], previous_states=states, is_first=tf.fill((B,), 1.0 if i == 0 else 0.0))
            states['a'] = actions[:, i]
        h_states_t0_to_H = [states['h']]
        z_states_prior_t0_to_H = [states['z']]
        a_t0_to_H = [states['a']]
        for j in range(timesteps_H):
            h = self.world_model.sequence_model(a=states['a'], h=states['h'], z=states['z'])
            h_states_t0_to_H.append(h)
            z, _ = self.world_model.dynamics_predictor(h=h)
            z_states_prior_t0_to_H.append(z)
            if use_sampled_actions_in_dream:
                a = actions[:, timesteps_burn_in + j]
            elif use_random_actions_in_dream:
                if isinstance(self.action_space, gym.spaces.Discrete):
                    a = tf.random.randint((B,), 0, self.action_space.n, tf.int64)
                    a = tf.one_hot(a, depth=self.action_space.n, dtype=tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32)
                else:
                    a = tf.random.uniform(shape=(B,) + self.action_space.shape, dtype=self.action_space.dtype)
            else:
                a, _ = self.actor(h=h, z=z)
            a_t0_to_H.append(a)
            states = {'h': h, 'z': z, 'a': a}
        h_states_t0_to_H_B = tf.stack(h_states_t0_to_H, axis=0)
        h_states_t0_to_HxB = tf.reshape(h_states_t0_to_H_B, shape=[-1] + h_states_t0_to_H_B.shape.as_list()[2:])
        z_states_prior_t0_to_H_B = tf.stack(z_states_prior_t0_to_H, axis=0)
        z_states_prior_t0_to_HxB = tf.reshape(z_states_prior_t0_to_H_B, shape=[-1] + z_states_prior_t0_to_H_B.shape.as_list()[2:])
        a_t0_to_H_B = tf.stack(a_t0_to_H, axis=0)
        o_dreamed_t0_to_HxB = self.world_model.decoder(h=h_states_t0_to_HxB, z=z_states_prior_t0_to_HxB)
        if self.world_model.symlog_obs:
            o_dreamed_t0_to_HxB = inverse_symlog(o_dreamed_t0_to_HxB)
        r_dreamed_t0_to_HxB, _ = self.world_model.reward_predictor(h=h_states_t0_to_HxB, z=z_states_prior_t0_to_HxB)
        r_dreamed_t0_to_HxB = inverse_symlog(r_dreamed_t0_to_HxB)
        c_dreamed_t0_to_HxB, _ = self.world_model.continue_predictor(h=h_states_t0_to_HxB, z=z_states_prior_t0_to_HxB)
        ret = {'h_states_t0_to_H_BxT': h_states_t0_to_H_B, 'z_states_prior_t0_to_H_BxT': z_states_prior_t0_to_H_B, 'observations_dreamed_t0_to_H_BxT': tf.reshape(o_dreamed_t0_to_HxB, [-1, B] + list(observations.shape)[2:]), 'rewards_dreamed_t0_to_H_BxT': tf.reshape(r_dreamed_t0_to_HxB, (-1, B)), 'continues_dreamed_t0_to_H_BxT': tf.reshape(c_dreamed_t0_to_HxB, (-1, B))}
        if use_sampled_actions_in_dream:
            key = 'actions_sampled_t0_to_H_BxT'
        elif use_random_actions_in_dream:
            key = 'actions_random_t0_to_H_BxT'
        else:
            key = 'actions_dreamed_t0_to_H_BxT'
        ret[key] = a_t0_to_H_B
        if isinstance(self.action_space, gym.spaces.Discrete):
            ret[re.sub('^actions_', 'actions_ints_', key)] = tf.argmax(a_t0_to_H_B, axis=-1)
        return ret