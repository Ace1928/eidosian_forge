import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
def _ssm_command_waiter(self, document_name: str, parameters: Dict[str, List[str]], node_id: str, retry_failed: bool=True) -> Dict[str, Any]:
    """wait for SSM command to complete on all cluster nodes"""
    response = self._send_command_to_node(document_name, parameters, node_id)
    command_id = response['Command']['CommandId']
    cloudwatch_config = self.provider_config['cloudwatch']
    agent_retryer_config = cloudwatch_config.get(CloudwatchConfigType.AGENT.value).get('retryer', {})
    max_attempts = agent_retryer_config.get('max_attempts', 120)
    delay_seconds = agent_retryer_config.get('delay_seconds', 30)
    num_attempts = 0
    cmd_invocation_res = {}
    while True:
        num_attempts += 1
        logger.debug('Listing SSM command ID {} invocations on node {}'.format(command_id, node_id))
        response = self.ssm_client.list_command_invocations(CommandId=command_id, InstanceId=node_id)
        cmd_invocations = response['CommandInvocations']
        if not cmd_invocations:
            logger.debug('SSM Command ID {} invocation does not exist. If the command was just started, it may take a few seconds to register.'.format(command_id))
        else:
            if len(cmd_invocations) > 1:
                logger.warning('Expected to find 1 SSM command invocation with ID {} on node {} but found {}: {}'.format(command_id, node_id, len(cmd_invocations), cmd_invocations))
            cmd_invocation = cmd_invocations[0]
            if cmd_invocation['Status'] == 'Success':
                logger.debug('SSM Command ID {} completed successfully.'.format(command_id))
                cmd_invocation_res[node_id] = True
                break
            if num_attempts >= max_attempts:
                logger.error('Max attempts for command {} exceeded on node {}'.format(command_id, node_id))
                raise botocore.exceptions.WaiterError(name='ssm_waiter', reason='Max attempts exceeded', last_response=cmd_invocation)
            if cmd_invocation['Status'] == 'Failed':
                logger.debug(f'SSM Command ID {command_id} failed.')
                if retry_failed:
                    logger.debug(f'Retrying in {delay_seconds} seconds.')
                    response = self._send_command_to_node(document_name, parameters, node_id)
                    command_id = response['Command']['CommandId']
                    logger.debug('Sent SSM command ID {} to node {}'.format(command_id, node_id))
                else:
                    logger.debug(f'Ignoring Command ID {command_id} failure.')
                    cmd_invocation_res[node_id] = False
                    break
        time.sleep(delay_seconds)
    return cmd_invocation_res