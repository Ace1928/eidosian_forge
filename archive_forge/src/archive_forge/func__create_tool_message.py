import json
from typing import List, Sequence, Tuple
from langchain_core.agents import AgentAction
from langchain_core.messages import (
from langchain.agents.output_parsers.tools import ToolAgentAction
def _create_tool_message(agent_action: ToolAgentAction, observation: str) -> ToolMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return ToolMessage(tool_call_id=agent_action.tool_call_id, content=content, additional_kwargs={'name': agent_action.tool})