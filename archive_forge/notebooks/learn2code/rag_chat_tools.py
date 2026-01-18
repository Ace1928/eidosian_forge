#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A comprehensive abstraction layer for Hugging Face Transformers models,
with advanced tool use, analysis, vector storage, and retrieval-augmented generation integration.

Optimized for CPU performance and modular for extension.
"""

import logging, json, re
from typing import Any, Dict, Optional, Union, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Olmo2Config,
    Olmo2ForCausalLM,
    GenerationConfig,
    TextStreamer
)
from sentence_transformers import SentenceTransformer
import faiss
import ragflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CPU optimization
torch.set_num_threads(torch.get_num_threads())

# --- Model Wrapper and Analysis ---

class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        offload_folder: str = "offload",
    ):
        self.model_name = model_name
        self.config_overrides = config_overrides or {}
        # Force CPU device
        self.device = torch.device("cpu")
        self.offload_folder = offload_folder

        self.config: Optional[PretrainedConfig] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self._load_configuration()
        self._load_tokenizer()
        self._load_model()

    def _load_configuration(self) -> None:
        logger.info("Loading model configuration...")
        self.config = Olmo2Config.from_pretrained(self.model_name, **self.config_overrides)

    def _load_model(self) -> None:
        logger.info("Loading model with disk offloading...")
        self.model = Olmo2ForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            device_map={"": "cpu"},
            offload_folder=self.offload_folder,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info("Model loaded, set to eval mode.")

    def _load_tokenizer(self) -> None:
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if not self.tokenizer or not self.model:
            raise ValueError("Tokenizer and model must be loaded before generation.")
        generation_kwargs = generation_kwargs or {}
        logger.info(f"Tokenizing prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_config = GenerationConfig(max_length=max_length, **generation_kwargs)
        logger.info("Generating sequences...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                generation_config=gen_config,
                num_return_sequences=num_return_sequences,
            )
        logger.info("Decoding generated sequences...")
        outputs = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return outputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Any, Tuple]:
        if not self.model:
            raise ValueError("Model must be loaded before calling forward.")
        logger.info("Performing forward pass...")
        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        logger.debug(f"Forward pass outputs: {outputs}")
        return outputs

class DummyVectorDB:
    def __init__(self):
        self.documents = []
    def add_documents(self, texts, embeddings, metadatas):
        for text, emb, meta in zip(texts, embeddings, metadatas):
            self.documents.append({"text": text, "embedding": emb, "metadata": meta})

class EnhancedModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace DummyVectorDB with a real vector DB as needed
        self.vector_db = DummyVectorDB()

    def _analyze_response(self, response: str) -> Dict[str, Any]:
        # Real analysis would use NLP libraries. Placeholder example:
        analysis = {
            "sentiment": "neutral",
            "keywords": ["example", "analysis"],
            "grammar_issues": [],
            "logical_consistency": True,
        }
        return analysis

    def _vectorize_and_store(self, text: str, metadata: Dict[str, Any]):
        tokens = self.tokenizer(text, return_tensors="pt").input_ids.tolist()[0]
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        self.vector_db.add_documents([text], [embeddings], metadatas=[metadata])

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        responses = super().generate(prompt, max_length, num_return_sequences, generation_kwargs)
        for response in responses:
            analysis_results = self._analyze_response(response)
            metadata = {"prompt": prompt, "analysis": analysis_results}
            self._vectorize_and_store(response, metadata)
        return responses

# Tool parsing and functions
def try_parse_tool_calls(content: str):
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: {e}")
    if tool_calls:
        c = content[:offset].strip() if content[:offset].strip() else ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": content.strip()}

def get_current_temperature(location: str, unit: str = "celsius") -> Dict[str, Any]:
    return {"temperature": 26.1, "location": location, "unit": unit}

def get_temperature_date(location: str, date: str, unit: str = "celsius") -> Dict[str, Any]:
    return {"temperature": 25.9, "location": location, "date": date, "unit": unit}

def get_function_by_name(name: str):
    return {
        "get_current_temperature": get_current_temperature,
        "get_temperature_date": get_temperature_date
    }.get(name)

def chat_interface(model, tokenizer, tools: List[Any]):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print("\nChat Interface Ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        messages.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        output_text = tokenizer.batch_decode(outputs)[0][len(text):]

        parsed_message = try_parse_tool_calls(output_text)
        messages.append(parsed_message)

        if tool_calls := parsed_message.get("tool_calls"):
            for tool_call in tool_calls:
                fn_call = tool_call.get("function")
                if fn_call:
                    fn_name = fn_call["name"]
                    fn_args = fn_call["arguments"]
                    fn_result = json.dumps(get_function_by_name(fn_name)(**fn_args))
                    messages.append({"role": "tool", "name": fn_name, "content": fn_result})

        print("Assistant:", parsed_message.get("content"))

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    CONFIG_OVERRIDES = {}
    logger.info("Initializing Enhanced model wrapper on CPU with disk offloading...")
    enhanced_wrapper = EnhancedModelWrapper(
        model_name=MODEL_NAME,
        config_overrides=CONFIG_OVERRIDES,
        device="cpu",
        offload_folder="offload"
    )

    # Example generation and analysis
    prompt_text = "Hey, are you conscious? Can you talk to me?"
    logger.info(f"Generating and analyzing response for prompt: {prompt_text}")
    responses = enhanced_wrapper.generate(prompt=prompt_text, max_length=50)
    for i, response in enumerate(responses):
        logger.info(f"Response {i+1}: {response}")

    # Start chat interface with tool use
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )
    tools = [get_current_temperature, get_temperature_date]
    chat_interface(model, tokenizer, tools)
