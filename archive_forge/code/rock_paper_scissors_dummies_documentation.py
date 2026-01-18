import gymnasium as gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement
Play the move that would beat the last move of the opponent.