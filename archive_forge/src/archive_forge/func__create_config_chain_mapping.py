import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _create_config_chain_mapping(chain_builder, config_variables):
    mapping = {}
    for logical_name, config in config_variables.items():
        mapping[logical_name] = chain_builder.create_config_chain(instance_name=logical_name, env_var_names=config[1], config_property_names=config[0], default=config[2], conversion_func=config[3])
    return mapping