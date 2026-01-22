import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class AgentConnector(Connector):
    """Connector connecting user environments to RLlib policies.

    An agent connector transforms a list of agent data in AgentConnectorDataType
    format into a new list in the same AgentConnectorDataTypes format.
    The input API is designed so agent connectors can have access to all the
    agents assigned to a particular policy.

    AgentConnectorDataTypes can be used to specify arbitrary type of env data,

    Example:

        Represent a list of agent data from one env step() call.

        .. testcode::

            import numpy as np
            ac = AgentConnectorDataType(
                env_id="env_1",
                agent_id=None,
                data={
                    "agent_1": np.array([1, 2, 3]),
                    "agent_2": np.array([4, 5, 6]),
                }
            )

        Or a single agent data ready to be preprocessed.

        .. testcode::

            ac = AgentConnectorDataType(
                env_id="env_1",
                agent_id="agent_1",
                data=np.array([1, 2, 3]),
            )

        We can also adapt a simple stateless function into an agent connector by
        using register_lambda_agent_connector:

        .. testcode::

            import numpy as np
            from ray.rllib.connectors.agent.lambdas import (
                register_lambda_agent_connector
            )
            TimesTwoAgentConnector = register_lambda_agent_connector(
                "TimesTwoAgentConnector", lambda data: data * 2
            )

            # More complicated agent connectors can be implemented by extending this
            # AgentConnector class:

            class FrameSkippingAgentConnector(AgentConnector):
                def __init__(self, n):
                    self._n = n
                    self._frame_count = default_dict(str, default_dict(str, int))

                def reset(self, env_id: str):
                    del self._frame_count[env_id]

                def __call__(
                    self, ac_data: List[AgentConnectorDataType]
                ) -> List[AgentConnectorDataType]:
                    ret = []
                    for d in ac_data:
                        assert d.env_id and d.agent_id, "Skipping works per agent!"

                        count = self._frame_count[ac_data.env_id][ac_data.agent_id]
                        self._frame_count[ac_data.env_id][ac_data.agent_id] = (
                            count + 1
                        )

                        if count % self._n == 0:
                            ret.append(d)
                    return ret

    As shown, an agent connector may choose to emit an empty list to stop input
    observations from being further prosessed.
    """

    def reset(self, env_id: str):
        """Reset connector state for a specific environment.

        For example, at the end of an episode.

        Args:
            env_id: required. ID of a user environment. Required.
        """
        pass

    def on_policy_output(self, output: ActionConnectorDataType):
        """Callback on agent connector of policy output.

        This is useful for certain connectors, for example RNN state buffering,
        where the agent connect needs to be aware of the output of a policy
        forward pass.

        Args:
            ctx: Context for running this connector call.
            output: Env and agent IDs, plus data output from policy forward pass.
        """
        pass

    def __call__(self, acd_list: List[AgentConnectorDataType]) -> List[AgentConnectorDataType]:
        """Transform a list of data items from env before they reach policy.

        Args:
            ac_data: List of env and agent IDs, plus arbitrary data items from
                an environment or upstream agent connectors.

        Returns:
            A list of transformed data items in AgentConnectorDataType format.
            The shape of a returned list does not have to match that of the input list.
            An AgentConnector may choose to derive multiple outputs for a single piece
            of input data, for example multi-agent obs -> multiple single agent obs.
            Agent connectors may also choose to skip emitting certain inputs,
            useful for connectors such as frame skipping.
        """
        assert isinstance(acd_list, (list, tuple)), 'Input to agent connectors are list of AgentConnectorDataType.'
        return [self.transform(d) for d in acd_list]

    def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
        """Transform a single agent connector data item.

        Args:
            data: Env and agent IDs, plus arbitrary data item from a single agent
            of an environment.

        Returns:
            A transformed piece of agent connector data.
        """
        raise NotImplementedError