from typing import Optional
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict
An interface to the PettingZoo MARL environment library.

    See: https://github.com/Farama-Foundation/PettingZoo

    Inherits from MultiAgentEnv and exposes a given AEC
    (actor-environment-cycle) game from the PettingZoo project via the
    MultiAgentEnv public API.

    Note that the wrapper has the following important limitation:

    Environments are positive sum games (-> Agents are expected to cooperate
       to maximize reward). This isn't a hard restriction, it just that
       standard algorithms aren't expected to work well in highly competitive
       games.

    Also note that the earlier existing restriction of all agents having the same
    observation- and action spaces has been lifted. Different agents can now have
    different spaces and the entire environment's e.g. `self.action_space` is a Dict
    mapping agent IDs to individual agents' spaces. Same for `self.observation_space`.

    .. testcode::
        :skipif: True

        from pettingzoo.butterfly import prison_v3
        from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
        env = PettingZooEnv(prison_v3.env())
        obs, infos = env.reset()
        # only returns the observation for the agent which should be stepping
        print(obs)

    .. testoutput::

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

        obs, rewards, terminateds, truncateds, infos = env.step({
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

        print(terminateds)

    .. testoutput::

        {
            'prisoner_1': False, '__all__': False
        }

    .. testcode::
        :skipif: True

        print(truncateds)

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
    