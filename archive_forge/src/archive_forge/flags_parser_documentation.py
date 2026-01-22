import argparse
from tensorboard.util import grpc_util
Configures flags on the provided argument parser.

    Integration point for `tensorboard.program`'s subcommand system.

    Args:
      parser: An `argparse.ArgumentParser` to be mutated.
    