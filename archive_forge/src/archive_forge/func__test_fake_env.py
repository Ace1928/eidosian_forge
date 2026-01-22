from minerl.herobraine.hero import handlers
from typing import List
from minerl.herobraine.hero.handlers.translation import TranslationHandler
import time
from minerl.herobraine.env_specs.navigate_specs import Navigate
import coloredlogs
import logging
def _test_fake_env(env_spec, should_render=False):
    fake_env = env_spec.make(fake=True)
    assert fake_env.action_space == fake_env.task.action_space
    assert fake_env.observation_space == fake_env.observation_space
    assert fake_env._seed == None
    fake_env.seed(200)
    assert fake_env._seed == 200
    fake_obs = fake_env.reset()
    assert fake_obs in env_spec.observation_space
    for _ in range(100):
        fake_obs, _, _, fake_monitor = fake_env.step(fake_env.action_space.sample())
        if should_render:
            fake_env.render()
            time.sleep(0.1)
        assert fake_obs in env_spec.observation_space
        assert fake_monitor in env_spec.monitor_space