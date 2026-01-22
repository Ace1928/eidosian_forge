from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union, cast
from langchain_core.chat_sessions import ChatSession
from langchain_core.load.load import load
from langchain_community.chat_loaders.base import BaseChatLoader
@staticmethod
def _get_messages_from_llm_run(llm_run: 'Run') -> ChatSession:
    """
        Extract messages from a LangSmith LLM run.

        :param llm_run: The LLM run object.
        :return: ChatSession with the extracted messages.
        """
    if llm_run.run_type != 'llm':
        raise ValueError(f'Expected run of type llm. Got: {llm_run.run_type}')
    if 'messages' not in llm_run.inputs:
        raise ValueError(f"Run has no 'messages' inputs. Got {llm_run.inputs}")
    if not llm_run.outputs:
        raise ValueError('Cannot convert pending run')
    messages = load(llm_run.inputs)['messages']
    message_chunk = load(llm_run.outputs)['generations'][0]['message']
    return ChatSession(messages=messages + [message_chunk])