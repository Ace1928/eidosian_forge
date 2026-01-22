import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.utils import override
Builds the World-Model's decoder network depending on the obs space.