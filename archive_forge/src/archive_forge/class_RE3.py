from gymnasium.spaces import Box, Discrete, Space
import numpy as np
from typing import List, Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
@PublicAPI
class RE3(Exploration):
    """Random Encoder for Efficient Exploration.

    Implementation of:
    [1] State entropy maximization with random encoders for efficient
    exploration. Seo, Chen, Shin, Lee, Abbeel, & Lee, (2021).
    arXiv preprint arXiv:2102.09430.

    Estimates state entropy using a particle-based k-nearest neighbors (k-NN)
    estimator in the latent space. The state's latent representation is
    calculated using an encoder with randomly initialized parameters.

    The entropy of a state is considered as intrinsic reward and added to the
    environment's extrinsic reward for policy optimization.
    Entropy is calculated per batch, it does not take the distribution of
    the entire replay buffer into consideration.
    """

    def __init__(self, action_space: Space, *, framework: str, model: ModelV2, embeds_dim: int=128, encoder_net_config: Optional[ModelConfigDict]=None, beta: float=0.2, beta_schedule: str='constant', rho: float=0.1, k_nn: int=50, random_timesteps: int=10000, sub_exploration: Optional[FromConfigSpec]=None, **kwargs):
        """Initialize RE3.

        Args:
            action_space: The action space in which to explore.
            framework: Supports "tf", this implementation does not
                support torch.
            model: The policy's model.
            embeds_dim: The dimensionality of the observation embedding
                vectors in latent space.
            encoder_net_config: Optional model
                configuration for the encoder network, producing embedding
                vectors from observations. This can be used to configure
                fcnet- or conv_net setups to properly process any
                observation space.
            beta: Hyperparameter to choose between exploration and
                exploitation.
            beta_schedule: Schedule to use for beta decay, one of
                "constant" or "linear_decay".
            rho: Beta decay factor, used for on-policy algorithm.
            k_nn: Number of neighbours to set for K-NN entropy
                estimation.
            random_timesteps: The number of timesteps to act completely
                randomly (see [1]).
            sub_exploration: The config dict for the underlying Exploration
                to use (e.g. epsilon-greedy for DQN). If None, uses the
                FromSpecDict provided in the Policy's default config.

        Raises:
            ValueError: If the input framework is Torch.
        """
        if framework == 'torch':
            raise ValueError('This RE3 implementation does not support Torch.')
        super().__init__(action_space, model=model, framework=framework, **kwargs)
        self.beta = beta
        self.rho = rho
        self.k_nn = k_nn
        self.embeds_dim = embeds_dim
        if encoder_net_config is None:
            encoder_net_config = self.policy_config['model'].copy()
        self.encoder_net_config = encoder_net_config
        if sub_exploration is None:
            if isinstance(self.action_space, Discrete):
                sub_exploration = {'type': 'EpsilonGreedy', 'epsilon_schedule': {'type': 'PiecewiseSchedule', 'endpoints': [(0, 1.0), (random_timesteps + 1, 1.0), (random_timesteps + 2, 0.01)], 'outside_value': 0.01}}
            elif isinstance(self.action_space, Box):
                sub_exploration = {'type': 'OrnsteinUhlenbeckNoise', 'random_timesteps': random_timesteps}
            else:
                raise NotImplementedError
        self.sub_exploration = sub_exploration
        self._encoder_net = ModelCatalog.get_model_v2(self.model.obs_space, self.action_space, self.embeds_dim, model_config=self.encoder_net_config, framework=self.framework, name='encoder_net')
        if self.framework == 'tf':
            self._obs_ph = get_placeholder(space=self.model.obs_space, name='_encoder_obs')
            self._obs_embeds = tf.stop_gradient(self._encoder_net({SampleBatch.OBS: self._obs_ph})[0])
        self.exploration_submodule = from_config(cls=Exploration, config=self.sub_exploration, action_space=self.action_space, framework=self.framework, policy_config=self.policy_config, model=self.model, num_workers=self.num_workers, worker_index=self.worker_index)

    @override(Exploration)
    def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Union[int, TensorType], explore: bool=True):
        return self.exploration_submodule.get_exploration_action(action_distribution=action_distribution, timestep=timestep, explore=explore)

    @override(Exploration)
    def postprocess_trajectory(self, policy, sample_batch, tf_sess=None):
        """Calculate states' latent representations/embeddings.

        Embeddings are added to the SampleBatch object such that it doesn't
        need to be calculated during each training step.
        """
        if self.framework != 'torch':
            sample_batch = self._postprocess_tf(policy, sample_batch, tf_sess)
        else:
            raise ValueError('Not implemented for Torch.')
        return sample_batch

    def _postprocess_tf(self, policy, sample_batch, tf_sess):
        """Calculate states' embeddings and add it to SampleBatch."""
        if self.framework == 'tf':
            obs_embeds = tf_sess.run(self._obs_embeds, feed_dict={self._obs_ph: sample_batch[SampleBatch.OBS]})
        else:
            obs_embeds = tf.stop_gradient(self._encoder_net({SampleBatch.OBS: sample_batch[SampleBatch.OBS]})[0]).numpy()
        sample_batch[SampleBatch.OBS_EMBEDS] = obs_embeds
        return sample_batch