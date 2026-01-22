import threading
from typing import Any, Dict, List, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
def _get_anthropic_claude_token_cost(prompt_tokens: int, completion_tokens: int, model_id: Union[str, None]) -> float:
    """Get the cost of tokens for the Claude model."""
    if model_id not in MODEL_COST_PER_1K_INPUT_TOKENS:
        raise ValueError(f'Unknown model: {model_id}. Please provide a valid Anthropic model name.Known models are: ' + ', '.join(MODEL_COST_PER_1K_INPUT_TOKENS.keys()))
    return prompt_tokens / 1000 * MODEL_COST_PER_1K_INPUT_TOKENS[model_id] + completion_tokens / 1000 * MODEL_COST_PER_1K_OUTPUT_TOKENS[model_id]