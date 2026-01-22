from typing import TYPE_CHECKING, Any, Dict, Generator, List, Mapping, Optional, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
def _stream_generate(self, model: Union['RESTfulGenerateModelHandle', 'RESTfulChatModelHandle'], prompt: str, run_manager: Optional[CallbackManagerForLLMRun]=None, generate_config: Optional['LlamaCppGenerateConfig']=None) -> Generator[str, None, None]:
    """
        Args:
            prompt: The prompt to use for generation.
            model: The model used for generation.
            stop: Optional list of stop words to use when generating.
            generate_config: Optional dictionary for the configuration used for
                generation.

        Yields:
            A string token.
        """
    streaming_response = model.generate(prompt=prompt, generate_config=generate_config)
    for chunk in streaming_response:
        if isinstance(chunk, dict):
            choices = chunk.get('choices', [])
            if choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    token = choice.get('text', '')
                    log_probs = choice.get('logprobs')
                    if run_manager:
                        run_manager.on_llm_new_token(token=token, verbose=self.verbose, log_probs=log_probs)
                    yield token