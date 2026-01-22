import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict
import requests
from huggingface_hub import HfFolder, hf_hub_download, list_spaces
from ..models.auto import AutoTokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
from .python_interpreter import evaluate
class HfAgent(Agent):
    """
    Agent that uses an inference endpoint to generate code.

    Args:
        url_endpoint (`str`):
            The name of the url endpoint to use.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from transformers import HfAgent

    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(self, url_endpoint, token=None, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        self.url_endpoint = url_endpoint
        if token is None:
            self.token = f'Bearer {HfFolder().get_token()}'
        elif token.startswith('Bearer') or token.startswith('Basic'):
            self.token = token
        else:
            self.token = f'Bearer {token}'
        super().__init__(chat_prompt_template=chat_prompt_template, run_prompt_template=run_prompt_template, additional_tools=additional_tools)

    def generate_one(self, prompt, stop):
        headers = {'Authorization': self.token}
        inputs = {'inputs': prompt, 'parameters': {'max_new_tokens': 200, 'return_full_text': False, 'stop': stop}}
        response = requests.post(self.url_endpoint, json=inputs, headers=headers)
        if response.status_code == 429:
            logger.info('Getting rate-limited, waiting a tiny bit before trying again.')
            time.sleep(1)
            return self._generate_one(prompt)
        elif response.status_code != 200:
            raise ValueError(f'Error {response.status_code}: {response.json()}')
        result = response.json()[0]['generated_text']
        for stop_seq in stop:
            if result.endswith(stop_seq):
                return result[:-len(stop_seq)]
        return result