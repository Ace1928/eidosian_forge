import logging
import os
import stat
from ray.autoscaler._private.aliyun.utils import AcsClient
def _get_or_create_security_group(config):
    cli = _client(config)
    security_groups = cli.describe_security_groups(vpc_id=config['provider']['vpc_id'])
    if security_groups is not None and len(security_groups) > 0:
        config['provider']['security_group_id'] = security_groups[0]['SecurityGroupId']
        return config
    security_group_id = cli.create_security_group(vpc_id=config['provider']['vpc_id'])
    for rule in config['provider'].get('security_group_rule', {}):
        cli.authorize_security_group(security_group_id=security_group_id, port_range=rule['port_range'], source_cidr_ip=rule['source_cidr_ip'], ip_protocol=rule['ip_protocol'])
    config['provider']['security_group_id'] = security_group_id
    return