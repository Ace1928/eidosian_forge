import random
from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.schedulers import PopulationBasedTraining
Example of using PBT with RLlib.

Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).

Note that Tune in general does not need 8 GPUs, and this is just a more
computationally demanding example.
