import uuid
import gymnasium as gym
from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.env.external_env import ExternalEnv, _ExternalEnvEpisode
from ray.rllib.utils.typing import MultiAgentDict
Record the end of an episode.

        Args:
            episode_id: Episode id returned from start_episode().
            observation_dict: Current environment observation.
        