from typing import Any, Callable, Type
import numpy as np
import tree  # dm_tree
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
class LambdaAgentConnector(AgentConnector):

    def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
        return AgentConnectorDataType(ac_data.env_id, ac_data.agent_id, fn(ac_data.data))

    def to_state(self):
        return (name, None)

    @staticmethod
    def from_state(ctx: ConnectorContext, params: Any):
        return LambdaAgentConnector(ctx)