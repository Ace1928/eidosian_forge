from __future__ import annotations
import functools
import logging
from collections import defaultdict
from typing import (
from langsmith import run_helpers
def _reduce_choices(choices: List[Choice]) -> dict:
    reversed_choices = list(reversed(choices))
    message: Dict[str, Any] = {'role': 'assistant', 'content': ''}
    for c in reversed_choices:
        if c.delta.role:
            message['role'] = c.delta.role
            break
    tool_calls: DefaultDict[int, List[ChoiceDeltaToolCall]] = defaultdict(list)
    for c in choices:
        if c.delta.content:
            message['content'] += c.delta.content
        if c.delta.function_call:
            if not message.get('function_call'):
                message['function_call'] = {'name': '', 'arguments': ''}
            if c.delta.function_call.name:
                message['function_call']['name'] += c.delta.function_call.name
            if c.delta.function_call.arguments:
                message['function_call']['arguments'] += c.delta.function_call.arguments
        if c.delta.tool_calls:
            for tool_call in c.delta.tool_calls:
                tool_calls[c.index].append(tool_call)
    if tool_calls:
        message['tool_calls'] = [None for _ in tool_calls.keys()]
        for index, tool_call_chunks in tool_calls.items():
            message['tool_calls'][index] = {'index': index, 'id': next((c.id for c in tool_call_chunks if c.id), None), 'type': next((c.type for c in tool_call_chunks if c.type), None)}
            for chunk in tool_call_chunks:
                if chunk.function:
                    if not message['tool_calls'][index].get('function'):
                        message['tool_calls'][index]['function'] = {'name': '', 'arguments': ''}
                    if chunk.function.name:
                        message['tool_calls'][index]['function']['name'] += chunk.function.name
                    if chunk.function.arguments:
                        message['tool_calls'][index]['function']['arguments'] += chunk.function.arguments
    return {'index': choices[0].index, 'finish_reason': next((c.finish_reason for c in reversed_choices if c.finish_reason), None), 'message': message}