import argparse
import os
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls
class MyPrintLogger(Logger):
    """Logs results by simply printing out everything."""

    def _init(self):
        print('Initializing ...')
        self.prefix = self.config.get('logger_config').get('prefix')

    def on_result(self, result: dict):
        print(f'{self.prefix}: {result}')

    def close(self):
        print('Closing')

    def flush(self):
        print('Flushing ;)', flush=True)