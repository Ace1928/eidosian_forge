from typing import Optional
import torch
Metadata for input sequences. Used in PagedAttention.

    Args:
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
        kv_cache_dtype: Data type to store kv cache.
    