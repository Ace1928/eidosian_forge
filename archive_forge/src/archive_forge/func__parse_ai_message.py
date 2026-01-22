import json
from json import JSONDecodeError
from typing import List, Union
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, Generation
from langchain.agents.agent import AgentOutputParser
@staticmethod
def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f'Expected an AI message got {type(message)}')
    function_call = message.additional_kwargs.get('function_call', {})
    if function_call:
        function_name = function_call['name']
        try:
            if len(function_call['arguments'].strip()) == 0:
                _tool_input = {}
            else:
                _tool_input = json.loads(function_call['arguments'], strict=False)
        except JSONDecodeError:
            raise OutputParserException(f'Could not parse tool input: {function_call} because the `arguments` is not valid JSON.')
        if '__arg1' in _tool_input:
            tool_input = _tool_input['__arg1']
        else:
            tool_input = _tool_input
        content_msg = f'responded: {message.content}\n' if message.content else '\n'
        log = f'\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n'
        return AgentActionMessageLog(tool=function_name, tool_input=tool_input, log=log, message_log=[message])
    return AgentFinish(return_values={'output': message.content}, log=str(message.content))