import sys
from typing import (
import numpy as np
from gym import spaces
from gym.logger import warn
from gym.utils import seeding
class ActionWrapper(Wrapper):
    """Superclass of wrappers that can modify the action before :meth:`env.step`.

    If you would like to apply a function to the action before passing it to the base environment,
    you can simply inherit from :class:`ActionWrapper` and overwrite the method :meth:`action` to implement
    that transformation. The transformation defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.

    Let’s say you have an environment with action space of type :class:`gym.spaces.Box`, but you would only like
    to use a finite subset of actions. Then, you might want to implement the following wrapper::

        class DiscreteActions(gym.ActionWrapper):
            def __init__(self, env, disc_to_cont):
                super().__init__(env)
                self.disc_to_cont = disc_to_cont
                self.action_space = Discrete(len(disc_to_cont))

            def action(self, act):
                return self.disc_to_cont[act]

        if __name__ == "__main__":
            env = gym.make("LunarLanderContinuous-v2")
            wrapped_env = DiscreteActions(env, [np.array([1,0]), np.array([-1,0]),
                                                np.array([0,1]), np.array([0,-1])])
            print(wrapped_env.action_space)         #Discrete(4)


    Among others, Gym provides the action wrappers :class:`ClipAction` and :class:`RescaleAction`.
    """

    def step(self, action):
        """Runs the environment :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(self.action(action))

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError