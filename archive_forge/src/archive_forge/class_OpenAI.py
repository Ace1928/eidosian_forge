import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
class OpenAI:
    """An object that represents the OpenAI API."""

    def __init__(self, client, config, tokenizer=None, system_prompt: Optional[str]=None):
        """Create an `OpenAI` instance.

        This class supports the standard OpenAI API, the Azure OpeanAI API as
        well as compatible APIs that rely on the OpenAI client.

        Parameters
        ----------
        client
            An instance of the API's async client.
        config
            An instance of `OpenAIConfig`. Can be useful to specify some
            parameters that cannot be set by calling this class' methods.
        tokenizer
            The tokenizer associated with the model the client connects to.

        """
        self.client = client
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, prompt: Union[str, List[str]], max_tokens: Optional[int]=None, stop_at: Optional[Union[List[str], str]]=None, *, system_prompt: Optional[str]=None, temperature: Optional[float]=None, samples: Optional[int]=None) -> np.ndarray:
        """Call the OpenAI API to generate text.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        max_tokens
            The maximum number of tokens to generate
        stop_at
            A string or array of strings which, such that the generation stops
            when they are generated.
        system_prompt
            The content of the system message that precedes the user's prompt.
        temperature
            The value of the temperature used to sample tokens
        samples
            The number of completions to generate for each prompt
        stop_at
            Up to 4 words where the API will stop the completion.

        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if stop_at is None:
            stop_at = self.config.stop
        if temperature is None:
            temperature = self.config.temperature
        if samples is None:
            samples = self.config.n
        config = replace(self.config, max_tokens=max_tokens, temperature=temperature, n=samples, stop=stop_at)
        response, prompt_tokens, completion_tokens = generate_chat(prompt, system_prompt, self.client, config)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        return response

    def stream(self, *args, **kwargs):
        raise NotImplementedError('Streaming is currently not supported for the OpenAI API')

    def generate_choice(self, prompt: str, choices: List[str], max_tokens: Optional[int]=None, system_prompt: Optional[str]=None) -> str:
        """Call the OpenAI API to generate one of several choices.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        choices
            The list of strings between which we ask the model to choose
        max_tokens
            The maximum number of tokens to generate
        system_prompt
            The content of the system message that precedes the user's prompt.

        """
        if self.tokenizer is None:
            raise ValueError('You must initialize the `OpenAI` class with a tokenizer to use `outlines.generate.choice`')
        config = replace(self.config, max_tokens=max_tokens)
        greedy = False
        decoded: List[str] = []
        encoded_choices_left: List[List[int]] = [self.tokenizer.encode(word) for word in choices]
        while len(encoded_choices_left) > 0:
            max_tokens_left = max([len(tokens) for tokens in encoded_choices_left])
            transposed_choices_left: List[Set] = [{item for item in subset if item is not None} for subset in zip_longest(*encoded_choices_left)]
            if not greedy:
                mask = build_optimistic_mask(transposed_choices_left)
            else:
                mask = {}
                for token in transposed_choices_left[0]:
                    mask[token] = 100
            if len(mask) == 0:
                break
            config = replace(config, logit_bias=mask, max_tokens=max_tokens_left)
            response, prompt_tokens, completion_tokens = generate_chat(prompt, system_prompt, self.client, config)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            encoded_response = self.tokenizer.encode(response)
            if encoded_response in encoded_choices_left:
                decoded.append(response)
                break
            else:
                encoded_response, encoded_choices_left = find_response_choices_intersection(encoded_response, encoded_choices_left)
                if len(encoded_response) == 0:
                    greedy = True
                    continue
                else:
                    decoded.append(''.join(self.tokenizer.decode(encoded_response)))
                    if len(encoded_choices_left) == 1:
                        choice_left = self.tokenizer.decode(encoded_choices_left[0])
                        decoded.append(choice_left)
                        break
                    greedy = False
                prompt = prompt + ''.join(decoded)
        choice = ''.join(decoded)
        return choice

    def generate_json(self):
        """Call the OpenAI API to generate a JSON object."""
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__ + ' API'

    def __repr__(self):
        return str(self.config)