import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def upgrade_policy(args=None, conf=None):
    logging.basicConfig(level=logging.WARN)
    if conf is None:
        conf = cfg.CONF
    conf.register_cli_opts(GENERATOR_OPTS + RULE_OPTS + UPGRADE_OPTS)
    conf.register_opts(GENERATOR_OPTS + RULE_OPTS + UPGRADE_OPTS)
    conf(args)
    _check_for_namespace_opt(conf)
    with open(conf.policy, 'r') as input_data:
        policies = policy.parse_file_contents(input_data.read())
    default_policies = get_policies_dict(conf.namespace)
    _upgrade_policies(policies, default_policies)
    if conf.output_file:
        with open(conf.output_file, 'w') as fh:
            if conf.format == 'yaml':
                yaml.safe_dump(policies, fh, default_flow_style=False)
            elif conf.format == 'json':
                LOG.warning(policy.WARN_JSON)
                jsonutils.dump(policies, fh, indent=4)
    elif conf.format == 'yaml':
        sys.stdout.write(yaml.safe_dump(policies, default_flow_style=False))
    elif conf.format == 'json':
        LOG.warning(policy.WARN_JSON)
        sys.stdout.write(jsonutils.dumps(policies, indent=4))