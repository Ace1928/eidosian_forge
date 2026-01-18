import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
@override(VectorEnv)
def vector_step(self, actions):
    self.ts += 1
    obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = ([], [], [], [], [])
    for i in range(self.num_envs):
        obs, rew, terminated, truncated, info = self.env.step(actions[i])
        if self.ts >= self.episode_len:
            truncated = True
        obs_batch.append(obs)
        rew_batch.append(rew)
        terminated_batch.append(terminated)
        truncated_batch.append(truncated)
        info_batch.append(info)
        if terminated or truncated:
            remaining = self.num_envs - (i + 1)
            obs_batch.extend([obs for _ in range(remaining)])
            rew_batch.extend([rew for _ in range(remaining)])
            terminated_batch.extend([terminated for _ in range(remaining)])
            truncated_batch.extend([truncated for _ in range(remaining)])
            info_batch.extend([info for _ in range(remaining)])
            break
    return (obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch)