import json
from typing import List, Sequence, Tuple
from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage
def _convert_agent_action_to_messages(agent_action: AgentAction, observation: str) -> List[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return list(agent_action.message_log) + [_create_function_message(agent_action, observation)]
    else:
        return [AIMessage(content=agent_action.log)]