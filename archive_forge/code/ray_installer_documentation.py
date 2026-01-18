import logging
import subprocess
from ray.autoscaler._private.updater import NodeUpdater
from ray.autoscaler._private.util import with_envs, with_head_node_ip
from ray.autoscaler.node_provider import NodeProvider as NodeProviderV1
from ray.autoscaler.v2.instance_manager.config import NodeProviderConfig
from ray.core.generated.instance_manager_pb2 import Instance

        Install ray on the target instance synchronously.
        