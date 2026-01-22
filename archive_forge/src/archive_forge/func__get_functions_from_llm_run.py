from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union, cast
from langchain_core.chat_sessions import ChatSession
from langchain_core.load.load import load
from langchain_community.chat_loaders.base import BaseChatLoader
@staticmethod
def _get_functions_from_llm_run(llm_run: 'Run') -> Optional[List[Dict]]:
    """
        Extract functions from a LangSmith LLM run if they exist.

        :param llm_run: The LLM run object.
        :return: Functions from the run or None.
        """
    if llm_run.run_type != 'llm':
        raise ValueError(f'Expected run of type llm. Got: {llm_run.run_type}')
    return (llm_run.extra or {}).get('invocation_params', {}).get('functions')