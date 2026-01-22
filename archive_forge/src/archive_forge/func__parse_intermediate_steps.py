from __future__ import annotations
import json
from json import JSONDecodeError
from time import sleep
from typing import (
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
def _parse_intermediate_steps(self, intermediate_steps: List[Tuple[OpenAIAssistantAction, str]]) -> dict:
    last_action, last_output = intermediate_steps[-1]
    run = self._wait_for_run(last_action.run_id, last_action.thread_id)
    required_tool_call_ids = {tc.id for tc in run.required_action.submit_tool_outputs.tool_calls}
    tool_outputs = [{'output': str(output), 'tool_call_id': action.tool_call_id} for action, output in intermediate_steps if action.tool_call_id in required_tool_call_ids]
    submit_tool_outputs = {'tool_outputs': tool_outputs, 'run_id': last_action.run_id, 'thread_id': last_action.thread_id}
    return submit_tool_outputs