from typing import List, Optional, Union
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.lora.request import LoRARequest
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
def _add_request(self, prompt: Optional[str], sampling_params: SamplingParams, prompt_token_ids: Optional[List[int]], lora_request: Optional[LoRARequest]=None, prefix_pos: Optional[int]=None) -> None:
    request_id = str(next(self.request_counter))
    self.llm_engine.add_request(request_id, prompt, sampling_params, prompt_token_ids, lora_request=lora_request, prefix_pos=prefix_pos)