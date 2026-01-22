from typing import List, Optional, Sequence
import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
class FindCaveEnvSpec(BasaltBaseEnvSpec):
    """
.. image:: ../assets/basalt/caves1_0-05.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/caves3_0-30.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/caves4_0-30.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/caves5_0-30.gif
  :scale: 100 %
  :alt:

After spawning in a plains biome, explore and find a cave. When inside a cave, end
the episode by setting the "ESC" action to 1.

You are not allowed to dig down from the surface to find a cave.
"""

    def __init__(self):
        super().__init__(name='MineRLBasaltFindCave-v0', demo_server_experiment_name='findcaves', max_episode_steps=3 * MINUTE, preferred_spawn_biome='plains', inventory=[])