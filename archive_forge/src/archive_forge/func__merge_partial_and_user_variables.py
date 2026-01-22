from typing import Any, Dict, List, Tuple
from langchain_core.agents import AgentAction
from langchain_core.prompts.chat import ChatPromptTemplate
def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
    intermediate_steps = kwargs.pop('intermediate_steps')
    kwargs['agent_scratchpad'] = self._construct_agent_scratchpad(intermediate_steps)
    return kwargs