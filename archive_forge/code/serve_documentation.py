import argparse
import json
from typing import AsyncGenerator
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor
Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - schema: the JSON schema to use for the generation (if regex is not provided).
    - regex: the regex to use for the generation (if schema is not provided).
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    