from typing import List, Optional, Sequence
import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
class MakeWaterfallEnvSpec(BasaltBaseEnvSpec):
    """
.. image:: ../assets/basalt/waterfall0_0-05.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/waterfall2_0-30.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/waterfall6_0-30.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/waterfall8_0-30.gif
  :scale: 100 %
  :alt:

After spawning in an extreme hills biome, use your waterbucket to make a beautiful waterfall.
Then take an aesthetic "picture" of it by moving to a good location, positioning
player's camera to have a nice view of the waterfall, and ending the episode by
setting "ESC" action to 1.
"""

    def __init__(self):
        super().__init__(name='MineRLBasaltMakeWaterfall-v0', demo_server_experiment_name='waterfall', max_episode_steps=5 * MINUTE, preferred_spawn_biome='extreme_hills', inventory=[dict(type='water_bucket', quantity=1), dict(type='cobblestone', quantity=20), dict(type='stone_shovel', quantity=1), dict(type='stone_pickaxe', quantity=1)])