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
class LocalAgent(Agent):
    """
    Agent that uses a local model and tokenizer to generate code.

    Args:
        model ([`PreTrainedModel`]):
            The model to use for the agent.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer to use for the agent.
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LocalAgent

    checkpoint = "bigcode/starcoder"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    agent = LocalAgent(model, tokenizer)
    agent.run("Draw me a picture of rivers and lakes.")
    ```
    """

    def __init__(self, model, tokenizer, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        self.model = model
        self.tokenizer = tokenizer
        super().__init__(chat_prompt_template=chat_prompt_template, run_prompt_template=run_prompt_template, additional_tools=additional_tools)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Convenience method to build a `LocalAgent` from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The name of a repo on the Hub or a local path to a folder containing both model and tokenizer.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments passed along to [`~PreTrainedModel.from_pretrained`].

        Example:

        ```py
        import torch
        from transformers import LocalAgent

        agent = LocalAgent.from_pretrained("bigcode/starcoder", device_map="auto", torch_dtype=torch.bfloat16)
        agent.run("Draw me a picture of rivers and lakes.")
        ```
        """
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model, tokenizer)

    @property
    def _model_device(self):
        if hasattr(self.model, 'hf_device_map'):
            return list(self.model.hf_device_map.values())[0]
        for param in self.model.parameters():
            return param.device

    def generate_one(self, prompt, stop):
        encoded_inputs = self.tokenizer(prompt, return_tensors='pt').to(self._model_device)
        src_len = encoded_inputs['input_ids'].shape[1]
        stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop, self.tokenizer)])
        outputs = self.model.generate(encoded_inputs['input_ids'], max_new_tokens=200, stopping_criteria=stopping_criteria)
        result = self.tokenizer.decode(outputs[0].tolist()[src_len:])
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[:-len(stop_seq)]
        return result