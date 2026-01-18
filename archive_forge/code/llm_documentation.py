from typing import List, Optional, Union
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.lora.request import LoRARequest
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        