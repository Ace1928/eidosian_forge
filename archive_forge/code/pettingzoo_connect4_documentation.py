import copy
from typing import Dict, Any
from pettingzoo import AECEnv
from pettingzoo.classic.connect_four_v3 import raw_env as connect_four_v3
from ray.rllib.env.multi_agent_env import MultiAgentEnv
An interface to the PettingZoo MARL environment library.
    See: https://github.com/Farama-Foundation/PettingZoo
    Inherits from MultiAgentEnv and exposes a given AEC
    (actor-environment-cycle) game from the PettingZoo project via the
    MultiAgentEnv public API.
    Note that the wrapper has some important limitations:
    1. All agents have the same action_spaces and observation_spaces.
       Note: If, within your aec game, agents do not have homogeneous action /
       observation spaces, apply SuperSuit wrappers
       to apply padding functionality: https://github.com/Farama-Foundation/
       SuperSuit#built-in-multi-agent-only-functions
    2. Environments are positive sum games (-> Agents are expected to cooperate
       to maximize reward). This isn't a hard restriction, it just that
       standard algorithms aren't expected to work well in highly competitive
       games.

    .. testcode::
        :skipif: True

        from pettingzoo.butterfly import prison_v3
        from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
        env = PettingZooEnv(prison_v3.env())
        obs = env.reset()
        print(obs)

    .. testoutput::

        # only returns the observation for the agent which should be stepping
        {
            'prisoner_0': array([[[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                ...,
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]], dtype=uint8)
        }

    .. testcode::
        :skipif: True

        obs, rewards, dones, infos = env.step({
                        "prisoner_0": 1
                    })
        # only returns the observation, reward, info, etc, for
        # the agent who's turn is next.
        print(obs)

    .. testoutput::

        {
            'prisoner_1': array([[[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                ...,
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]], dtype=uint8)
        }

    .. testcode::
        :skipif: True

        print(rewards)

    .. testoutput::

        {
            'prisoner_1': 0
        }

    .. testcode::
        :skipif: True

        print(dones)

    .. testoutput::

        {
            'prisoner_1': False, '__all__': False
        }

    .. testcode::
        :skipif: True

        print(infos)

    .. testoutput::

        {
            'prisoner_1': {'map_tuple': (1, 0)}
        }
    