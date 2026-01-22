import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import logging
import os
import sys
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def _on_step(self) -> bool:
    if self.model_save_freq > 0:
        if self.model_save_path is not None:
            if self.n_calls % self.model_save_freq == 0:
                self.save_model()
    return True