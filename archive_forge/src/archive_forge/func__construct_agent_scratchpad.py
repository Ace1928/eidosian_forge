from typing import Any, Dict, List, Tuple
from langchain_core.agents import AgentAction
from langchain_core.prompts.chat import ChatPromptTemplate
def _construct_agent_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
    if len(intermediate_steps) == 0:
        return ''
    thoughts = ''
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f'\nObservation: {observation}\nThought: '
    return f"This was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n{thoughts}"