import argparse
import time
from ray import train, tune
from ray.tune.logger import LoggerCallback
class TestLoggerCallback(LoggerCallback):

    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f'TestLogger for trial {trial}: {result}')