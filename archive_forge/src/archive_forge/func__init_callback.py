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
def _init_callback(self) -> None:
    d = {}
    if 'algo' not in d:
        d['algo'] = type(self.model).__name__
    for key in self.model.__dict__:
        if key in wandb.config:
            continue
        if type(self.model.__dict__[key]) in [float, int, str]:
            d[key] = self.model.__dict__[key]
        else:
            d[key] = str(self.model.__dict__[key])
    if self.gradient_save_freq > 0:
        wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log=self.log)
    wandb.config.setdefaults(d)