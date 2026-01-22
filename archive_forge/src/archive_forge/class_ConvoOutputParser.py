from __future__ import annotations
from typing import Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
class ConvoOutputParser(AgentOutputParser):
    """Output parser for the conversational agent."""
    format_instructions: str = FORMAT_INSTRUCTIONS
    'Default formatting instructions'

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into an AgentAction or AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        try:
            response = parse_json_markdown(text)
            if 'action' in response and 'action_input' in response:
                action, action_input = (response['action'], response['action_input'])
                if action == 'Final Answer':
                    return AgentFinish({'output': action_input}, text)
                else:
                    return AgentAction(action, action_input, text)
            else:
                raise OutputParserException(f"Missing 'action' or 'action_input' in LLM output: {text}")
        except Exception as e:
            raise OutputParserException(f'Could not parse LLM output: {text}') from e

    @property
    def _type(self) -> str:
        return 'conversational_chat'