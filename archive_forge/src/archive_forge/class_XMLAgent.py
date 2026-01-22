from typing import Any, List, Sequence, Tuple, Union
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain.agents.agent import BaseSingleActionAgent
from langchain.agents.format_scratchpad import format_xml
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.agents.xml.prompt import agent_instructions
from langchain.chains.llm import LLMChain
from langchain.tools.render import ToolsRenderer, render_text_description
@deprecated('0.1.0', alternative='create_xml_agent', removal='0.2.0')
class XMLAgent(BaseSingleActionAgent):
    """Agent that uses XML tags.

    Args:
        tools: list of tools the agent can choose from
        llm_chain: The LLMChain to call to predict the next action

    Examples:

        .. code-block:: python

            from langchain.agents import XMLAgent
            from langchain

            tools = ...
            model =


    """
    tools: List[BaseTool]
    'List of tools this agent has access to.'
    llm_chain: LLMChain
    'Chain to use to predict action.'

    @property
    def input_keys(self) -> List[str]:
        return ['input']

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        base_prompt = ChatPromptTemplate.from_template(agent_instructions)
        return base_prompt + AIMessagePromptTemplate.from_template('{intermediate_steps}')

    @staticmethod
    def get_default_output_parser() -> XMLAgentOutputParser:
        return XMLAgentOutputParser()

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks=None, **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        log = ''
        for action, observation in intermediate_steps:
            log += f'<tool>{action.tool}</tool><tool_input>{action.tool_input}</tool_input><observation>{observation}</observation>'
        tools = ''
        for tool in self.tools:
            tools += f'{tool.name}: {tool.description}\n'
        inputs = {'intermediate_steps': log, 'tools': tools, 'question': kwargs['input'], 'stop': ['</tool_input>', '</final_answer>']}
        response = self.llm_chain(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks=None, **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        log = ''
        for action, observation in intermediate_steps:
            log += f'<tool>{action.tool}</tool><tool_input>{action.tool_input}</tool_input><observation>{observation}</observation>'
        tools = ''
        for tool in self.tools:
            tools += f'{tool.name}: {tool.description}\n'
        inputs = {'intermediate_steps': log, 'tools': tools, 'question': kwargs['input'], 'stop': ['</tool_input>', '</final_answer>']}
        response = await self.llm_chain.acall(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]