from __future__ import annotations
from typing import Any, List, Optional, cast
from langchain_text_splitters.base import TextSplitter, Tokenizer, split_text_on_tokens
def _initialize_chunk_configuration(self, *, tokens_per_chunk: Optional[int]) -> None:
    self.maximum_tokens_per_chunk = cast(int, self._model.max_seq_length)
    if tokens_per_chunk is None:
        self.tokens_per_chunk = self.maximum_tokens_per_chunk
    else:
        self.tokens_per_chunk = tokens_per_chunk
    if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
        raise ValueError(f"The token limit of the models '{self.model_name}' is: {self.maximum_tokens_per_chunk}. Argument tokens_per_chunk={self.tokens_per_chunk} > maximum token limit.")