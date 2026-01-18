from typing import Optional
import gymnasium as gym
import tree  # pip install dm_tree
from ray.rllib.algorithms.dreamerv3.tf.models.components.continue_predictor import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.dynamics_predictor import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.representation_layer import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.reward_predictor import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.sequence_model import (
from ray.rllib.algorithms.dreamerv3.utils import get_gru_units
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import symlog
Performs a forward step for training.

        1) Forwards all observations [B, T, ...] through the encoder network to yield
        o_processed[B, T, ...].
        2) Uses initial state (h0/z^0/a0[B, 0, ...]) and sequence model (RSSM) to
        compute the first internal state (h1 and z^1).
        3) Uses action a[B, 1, ...], z[B, 1, ...] and h[B, 1, ...] to compute the
        next h-state (h[B, 2, ...]), etc..
        4) Repeats 2) and 3) until t=T.
        5) Uses all h[B, T, ...] and z[B, T, ...] to compute predicted/reconstructed
        observations, rewards, and continue signals.
        6) Returns predictions from 5) along with all z-states z[B, T, ...] and
        the final h-state (h[B, ...] for t=T).

        Should we encounter is_first=True flags in the middle of a batch row (somewhere
        within an ongoing sequence of length T), we insert this world model's initial
        state again (zero-action, learned init h-state, and prior-computed z^) and
        simply continue (no zero-padding).

        Args:
            observations: The batch (B, T, ...) of observations to be passed through
                the encoder network to yield the inputs to the representation layer
                (which then can compute the posterior z-states).
            actions: The batch (B, T, ...) of actions to be used in combination with
                h-states and computed z-states to yield the next h-states.
            is_first: The batch (B, T) of `is_first` flags.
        