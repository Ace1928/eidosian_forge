from minerl.env.malmo import InstanceManager
import minerl
import time
import gym
import numpy as np
import logging
import coloredlogs
from minerl.herobraine.wrappers.vector_wrapper import Vectorized
from minerl.herobraine.env_specs.obtain_specs import ObtainDiamondDebug
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
from minerl.herobraine.wrappers.obfuscation_wrapper import Obfuscated
import minerl.herobraine.envs as envs
import minerl.herobraine
def test_wrapped_env(environment='MineRLObtainTest-v0', wrapped_env='MineRLObtainTestVector-v0'):
    env = gym.make(environment)
    env.seed(1)
    wenv = gym.make(wrapped_env)
    wenv.seed(1)
    for _ in range(2):
        env.reset()
        wenv.reset()
        total_reward = 0
        action = env.action_space.no_op()
        action['equip'] = 'red_flower'
        print(action)
        waction = wenv.env_spec.wrap_action(action)
        _, _, _, _ = env.step(action)
        _, _, _, _ = wenv.step(waction)
        obs, _, _, _ = env.step(env.action_space.no_op())
        wobs, _, _, _ = wenv.step(wenv.env_spec.wrap_action(env.action_space.no_op()))
        unwobsed = wenv.env_spec.unwrap_observation(wobs)
        del obs['pov']
        del unwobsed['pov']
        assert_equal_recursive(obs, unwobsed)
        for action in gen_obtain_debug_actions(env):
            for key, value in action.items():
                if isinstance(value, str) and value in reward_dict and (key not in ['equip']):
                    print('Action of {}:{} if successful gets {}'.format(key, value, reward_dict[value]))
            obs, reward, done, info = env.step(action)
            wobs, wreward, wdone, winfo = wenv.step(wenv.env_spec.wrap_action(action))
            assert reward == wreward
            assert done == wdone
            unwobsed = wenv.env_spec.unwrap_observation(wobs)
            del obs['pov']
            del unwobsed['pov']
            total_reward += reward
            if done:
                assert_equal_recursive(obs, unwobsed)
                break
        print('MISSION DONE')
        assert_equal_recursive(obs, unwobsed)