from __future__ import annotations
import json
from typing import Any, List, Literal, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
def _convert_agent_observation_to_messages(agent_action: AgentAction, observation: Any) -> Sequence[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [_create_function_message(agent_action, observation)]
    else:
        return [HumanMessage(content=observation)]