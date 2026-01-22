import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
class TextEnvironment:
    """
    The TextEnvironment enables interaction of a LLM with an environment using tools.
    """

    def __init__(self, model=None, tokenizer=None, tools=None, reward_fn=None, prompt=None, max_turns=4, max_tool_reponse=100, max_length=None, generation_kwargs=None):
        """
        Initialize TextEnvironment.

        Args:
            model (`PreTrainedModelWrapper`): The model to use for generation.
            tokenizer (`transformers.PreTrainedTokenizer`): The tokenizer to use for generation.
            tools (list): A list of tools to use for interaction.
            reward_fn (function): A function that takes a string and returns a reward.
            prompt (str): The base prompt to use for generation. Is prepended to the tasks.
            max_turns (Optional[int]): The maximum number of turns to allow.
            max_tool_response (Optional[int]): The maximum number of characters to allow in a tool response.
            max_length (Optional[int]): The maximum number of tokens to allow in an episode.
            generation_kwargs (Optional[dict]): A dictionary of keyword arguments to pass to the model's generate method.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        if isinstance(tools, dict):
            self.tools = tools
        else:
            self.tools = {tool.__class__.__name__: tool for tool in tools}
        self.reward_fn = reward_fn
        self.max_length = max_length
        self.request_token = '<request>'
        self.call_token = '<call>'
        self.response_token = '<response>'
        self.submit_token = '<submit>'
        self.max_turns = max_turns
        self.max_tool_response = max_tool_reponse
        if generation_kwargs is None:
            self.generation_kwargs = dict()
        else:
            self.generation_kwargs = generation_kwargs
        self.is_encoder_decoder = hasattr(self.model, 'is_encoder_decoder')
        self.current_device = extract_model_from_parallel(self.model).pretrained_model.device

    def run(self, queries, **rewards_kwargs):
        """
        Run the environment on a list of queries.

        Args:
            queries (list[str]): A list of queries to run the model in the environment on.
        """
        turns = 0
        queries = [self.prompt + task for task in queries]
        queries_tokens = [self.tokenizer(query, return_tensors='pt').input_ids[0].to(self.model.pretrained_model.device) for query in queries]
        histories = [TextHistory(q, qt, system=True) for q, qt in zip(queries, queries_tokens)]
        while any((not history.completed for history in histories)) and turns < self.max_turns:
            histories = self.generate(histories)
            histories = self.tasks_end_check(histories)
            for i in range(len(histories)):
                histories[i] = self.step(histories[i])
            histories = self.tasks_end_check(histories, model_turn=False)
            turns += 1
        self.compute_reward(histories, **rewards_kwargs)
        queries, responses, masks = map(list, zip(*[history.split_query_response_tokens() for history in histories]))
        rewards = [history.reward for history in histories]
        return (queries, responses, masks, rewards, histories)

    def step(self, history):
        """
        Step the environment forward one turn.

        Args:
            history (`TextHistory`): The history to step forward.
        """
        truncated, ended = self.task_end_check(history)
        if ended:
            history.complete(truncated=truncated)
        if history.completed:
            return history
        tool, query = self.parse_tool_call(history.last_text_segment)
        if tool is None or query is None:
            response = f'Unknown tool call: {history.last_text_segment}'
        else:
            if tool not in self.tools:
                response = f'Unknown tool {tool}.'
            try:
                response = self.tools[tool](query)
            except Exception as error:
                response = f'Tool error: {str(error)}'
        if len(response) > self.max_tool_response:
            response = response[:self.max_tool_response - 3] + '...'
        history.append_segment(response + self.response_token, self.tokenizer(response + self.response_token, return_tensors='pt').input_ids[0].to(self.model.pretrained_model.device), system=True)
        return history

    def parse_tool_call(self, text):
        """
        Parse request string. Expected format: <request><tool_name>query<call>
        """
        result = re.search(f'(?<={self.request_token}).*?(?={self.call_token})', text, re.DOTALL)
        if result is None:
            return (None, None)
        else:
            extracted_text = result.group()
        result = re.search('<(.*?)>', extracted_text)
        if result is None:
            return (None, None)
        else:
            tool = result.group(1)
        query = '>'.join(extracted_text.split('>')[1:])
        return (tool, query)

    def compute_reward(self, histories, **reward_kwargs):
        """
        Compute the reward for a list of histories.
        """
        rewards = self.reward_fn([history.last_text_segment for history in histories], **reward_kwargs)
        for history, reward in zip(histories, rewards):
            history.reward = reward
        return histories

    def generate(self, histories):
        """
        Generate responses for a list of histories.
        """
        active_histories = [i for i, history in enumerate(histories) if not history.completed]
        query_tensors = [histories[i].tokens for i in active_histories]
        response_tensors = self._generate_batched(query_tensors)
        response_texts = self.tokenizer.batch_decode(response_tensors)
        for i, response_text, response_tensor in zip(active_histories, response_texts, response_tensors):
            histories[i].append_segment(response_text, response_tensor, system=False)
        return histories

    def tasks_end_check(self, histories, model_turn=True):
        """
        Check if the current generation sequences have finished.
        """
        for history in histories:
            if not history.completed:
                truncated, ended = self.task_end_check(history, model_turn=model_turn)
                if ended:
                    history.complete(truncated=truncated)
        return histories

    def task_end_check(self, history, model_turn=True):
        """
        Check if the current generation sequence has finished.
        """
        truncated = False
        ended = False
        if history.completed:
            return (truncated, ended)
        if self.max_length is not None and len(self.tokenizer(history.text).input_ids[0]) > self.max_length:
            truncated = True
            ended = True
        elif self.tokenizer.eos_token in history.text:
            ended = True
        elif model_turn and (not (self.request_token in history.last_text_segment and self.call_token in history.last_text_segment or self.submit_token in history.last_text_segment)):
            ended = True
        elif self.submit_token in history.last_text_segment:
            ended = True
        return (truncated, ended)

    def _generate_batched(self, query_tensors, batch_size: int=16, pad_to_multiple_of: Optional[int]=None):
        """
        Generate responses for a list of query tensors.

        args:
            query_tensors (list[torch.Tensor]): A list of query tensors to generate responses for.
            batch_size (int): The batch size to use for generation.
            pad_to_multiple_of (int): The padding length to use for generation.
        """
        outputs = []
        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = 'left'
        batch_size = min(len(query_tensors), batch_size)
        for i in range(0, len(query_tensors), batch_size):
            end_index = min(len(query_tensors), i + batch_size)
            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {'input_ids': batch, 'attention_mask': batch_mask}
            padded_inputs = self.tokenizer.pad(inputs, padding=True, max_length=None, pad_to_multiple_of=pad_to_multiple_of, return_tensors='pt').to(self.current_device)
            stopping_criteria = StringStoppingCriteria([self.call_token, self.submit_token], self.tokenizer)
            self.generation_kwargs['stopping_criteria'] = StoppingCriteriaList([stopping_criteria])
            generations = extract_model_from_parallel(self.model).generate(**padded_inputs, **self.generation_kwargs)
            for generation, mask, generated_tokens in zip(generations, padded_inputs['attention_mask'], stopping_criteria.generated_tokens):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum():]
                else:
                    output = generation
                if not self.is_encoder_decoder:
                    output = output[mask.sum():]
                outputs.append(output[:generated_tokens])
        self.tokenizer.padding_side = padding_side_default
        return outputs