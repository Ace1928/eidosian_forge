from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key

        Step environment for one frame.

        If `action` is not None, assume it is a valid action and pass it to the environment.
        Otherwise read action from player (current keyboard/mouse state).

        If `override_if_human_input` is True, execeute action from the human player if they
        press any button or move mouse.

        The executed action will be added to the info dict as "taken_action".
        