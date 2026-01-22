import argparse
import json
from ray.tune.utils.serialization import TuneFunctionEncoder
from ray.train import CheckpointConfig
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.resources import json_to_resources
from ray.tune.utils.util import SafeFallbackEncoder
Creates a Trial object from parsing the spec.

    Args:
        spec: A resolved experiment specification. Arguments should
            The args here should correspond to the command line flags
            in ray.tune.experiment.config_parser.
        parser: An argument parser object from
            make_parser.
        trial_kwargs: Extra keyword arguments used in instantiating the Trial.

    Returns:
        A trial object with corresponding parameters to the specification.
    