import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from redis._parsers import CommandsParser, Encoder
from redis._parsers.helpers import parse_scan
from redis.backoff import default_backoff
from redis.client import CaseInsensitiveDict, PubSub, Redis
from redis.commands import READ_COMMANDS, RedisClusterCommands
from redis.commands.helpers import list_or_args
from redis.connection import ConnectionPool, DefaultParser, parse_url
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def _send_cluster_commands(self, stack, raise_on_error=True, allow_redirections=True):
    """
        Send a bunch of cluster commands to the redis cluster.

        `allow_redirections` If the pipeline should follow
        `ASK` & `MOVED` responses automatically. If set
        to false it will raise RedisClusterException.
        """
    attempt = sorted(stack, key=lambda x: x.position)
    is_default_node = False
    nodes = {}
    for c in attempt:
        while True:
            passed_targets = c.options.pop('target_nodes', None)
            if passed_targets and (not self._is_nodes_flag(passed_targets)):
                target_nodes = self._parse_target_nodes(passed_targets)
            else:
                target_nodes = self._determine_nodes(*c.args, node_flag=passed_targets)
                if not target_nodes:
                    raise RedisClusterException(f'No targets were found to execute {c.args} command on')
            if len(target_nodes) > 1:
                raise RedisClusterException(f'Too many targets for command {c.args}')
            node = target_nodes[0]
            if node == self.get_default_node():
                is_default_node = True
            node_name = node.name
            if node_name not in nodes:
                redis_node = self.get_redis_connection(node)
                try:
                    connection = get_connection(redis_node, c.args)
                except ConnectionError:
                    for n in nodes.values():
                        n.connection_pool.release(n.connection)
                    self.nodes_manager.initialize()
                    if is_default_node:
                        self.replace_default_node()
                    raise
                nodes[node_name] = NodeCommands(redis_node.parse_response, redis_node.connection_pool, connection)
            nodes[node_name].append(c)
            break
    node_commands = nodes.values()
    try:
        node_commands = nodes.values()
        for n in node_commands:
            n.write()
        for n in node_commands:
            n.read()
    finally:
        for n in nodes.values():
            n.connection_pool.release(n.connection)
    attempt = sorted((c for c in attempt if isinstance(c.result, ClusterPipeline.ERRORS_ALLOW_RETRY)), key=lambda x: x.position)
    if attempt and allow_redirections:
        self.reinitialize_counter += 1
        if self._should_reinitialized():
            self.nodes_manager.initialize()
            if is_default_node:
                self.replace_default_node()
        for c in attempt:
            try:
                c.result = super().execute_command(*c.args, **c.options)
            except RedisError as e:
                c.result = e
    response = []
    for c in sorted(stack, key=lambda x: x.position):
        if c.args[0] in self.cluster_response_callbacks:
            c.result = self.cluster_response_callbacks[c.args[0]](c.result, **c.options)
        response.append(c.result)
    if raise_on_error:
        self.raise_first_error(stack)
    return response