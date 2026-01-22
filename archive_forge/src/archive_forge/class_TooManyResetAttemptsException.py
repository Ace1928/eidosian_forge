import logging
import traceback
import gymnasium as gym
class TooManyResetAttemptsException(Exception):

    def __init__(self, max_attempts: int):
        super().__init__(f'Reached the maximum number of attempts ({max_attempts}) to reset an environment.')