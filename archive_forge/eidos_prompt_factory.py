import logging
import json
import os
from typing import Dict, Optional, Any, List, TYPE_CHECKING, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
import uuid
import time
from eidos_prompts import (
    ASSESSOR_SYSTEM_PROMPT_CONTENT,
    REFINEMENT_PLAN_SYSTEM_PROMPT_CONTENT,
)

logger = logging.getLogger(__name__)

class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def get_content(self) -> str:
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        pass


class TextPromptTemplate(PromptTemplate):
    """Concrete class for text-based prompt templates."""
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self._content = content
        self._metadata = metadata or {}
    def get_content(self) -> str:
        return self._content
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

class JSONPromptTemplate(PromptTemplate):
    """Concrete class for JSON-based prompt templates."""
    def __init__(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self._data = data
        self._metadata = metadata or {}
    def get_content(self) -> str:
        return json.dumps(self._data)
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

@dataclass
class PromptTemplateManagerConfig:
    """Configuration for the PromptTemplateManager."""
    template_dir: str = "prompt_templates"
    template_file_extension: str = ".txt"
    template_metadata_extension: str = ".json"
    use_json_templates: bool = False
    load_on_init: bool = True
    initial_max_tokens: int = 100
    max_single_response_tokens: int = 2000
    assessor_count: int = 3

class PromptTemplateManager:
    """Manages prompt templates and creates prompts."""

    def __init__(self, config: Optional[PromptTemplateManagerConfig] = None):
        """Initializes the PromptTemplateManager."""
        self.config = config or PromptTemplateManagerConfig()
        self.templates: Dict[str, PromptTemplate] = {}
        self._lock = threading.RLock()
        if self.config.load_on_init:
            self.load_templates()

    def _create_base_prompt(self, user_prompt: str) -> str:
        return f"<PROMPT_START>\n<USER_PROMPT>\n{user_prompt}\n</USER_PROMPT>\n"

    def _add_previous_assessments_to_prompt(
        self, prompt: str, previous_assessments: Optional[List[str]]
    ) -> str:
        if previous_assessments:
            prompt += "<PREVIOUS_ASSESSMENTS>\n"
            for i, assessment in enumerate(previous_assessments):
                prompt += f"<ASSESSMENT_{i + 1}>\n{assessment}\n</ASSESSMENT_{i + 1}>\n"
            prompt += "</PREVIOUS_ASSESSMENTS>\n"
        return prompt

    def _add_cycle_information_to_prompt(self, prompt: str, cycle: int) -> str:
        prompt += f"<CYCLE>\n{cycle}\n</CYCLE>\n"
        return prompt

    def _add_response_under_review_to_prompt(
        self, prompt: str, initial_response: str
    ) -> str:
        prompt += (
            f"<RESPONSE_UNDER_REVIEW>\n{initial_response}\n</RESPONSE_UNDER_REVIEW>\n"
        )
        return prompt

    def create_critique_prompt(
        self,
        user_prompt: str,
        initial_response: str,
        previous_assessments: Optional[List[str]],
        cycle: int,
    ) -> str:
        prompt = self._create_base_prompt(user_prompt)
        prompt = self._add_cycle_information_to_prompt(prompt, cycle)
        prompt = self._add_response_under_review_to_prompt(prompt, initial_response)
        prompt = self._add_previous_assessments_to_prompt(prompt, previous_assessments)
        prompt += "<CRITIQUE_INSTRUCTIONS>\nProvide specific, actionable feedback, building upon previous critiques.\n</CRITIQUE_INSTRUCTIONS>\n<PROMPT_END>"
        return prompt

    def create_refinement_plan_prompt(
        self, user_prompt: str, initial_response: str, assessments: List[str]
    ) -> str:
        prompt = self._create_base_prompt(user_prompt)
        prompt += self._add_response_under_review_to_prompt(prompt, initial_response)
        prompt += "<ASSESSMENTS>\n"
        for i, assessment in enumerate(assessments):
            prompt += f"<ASSESSMENT_{i + 1}>\n{assessment}\n</ASSESSMENT_{i + 1}>\n"
        prompt += "</ASSESSMENTS>\n"
        prompt += "<REFINEMENT_INSTRUCTIONS>\nFormulate a detailed plan for refining the response based on the provided assessments.\n</REFINEMENT_INSTRUCTIONS>\n<PROMPT_END>"
        return prompt

    def create_refined_response_prompt(
        self, user_prompt: str, initial_response: str, refinement_plan: str
    ) -> str:
        prompt = self._create_base_prompt(user_prompt)
        prompt += self._add_response_under_review_to_prompt(prompt, initial_response)
        prompt += f"<REFINEMENT_PLAN>\n{refinement_plan}\n</REFINEMENT_PLAN>\n"
        prompt += "<REFINEMENT_INSTRUCTIONS>\nRefine the initial response according to the provided plan.\n</REFINEMENT_INSTRUCTIONS>\n<PROMPT_END>"
        return prompt

    def _load_template(
        self, template_path: str, metadata_path: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """Loads a prompt template from a file."""
        try:
            metadata = self._load_metadata(metadata_path)
            with open(template_path, "r", encoding="utf-8") as f:
                if self.config.use_json_templates:
                    return JSONPromptTemplate(json.load(f), metadata)
                else:
                    return TextPromptTemplate(f.read(), metadata)
        except Exception as e:
            logger.error(
                f"🔥 Error loading template from {template_path}: {e}", exc_info=True
            )
            return None

    def _load_metadata(self, metadata_path: str) -> Optional[Dict[str, Any]]:
        """Loads metadata for a prompt template from a JSON file."""
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(
                    f"🔥 Error loading metadata from {metadata_path}: {e}",
                    exc_info=True,
                )
        return None

    def load_templates(self) -> None:
        """Loads all prompt templates from the configured directory."""
        with self._lock:
            if not os.path.exists(self.config.template_dir):
                logger.warning(
                    f"⚠️ Template directory '{self.config.template_dir}' does not exist."
                )
                return

            for filename in os.listdir(self.config.template_dir):
                if filename.endswith(self.config.template_file_extension):
                    template_id = os.path.splitext(filename)[0]
                    template_path = os.path.join(self.config.template_dir, filename)
                    metadata_path = os.path.join(
                        self.config.template_dir,
                        f"{template_id}{self.config.template_metadata_extension}",
                    )
                    template = self._load_template(
                        template_path,
                        metadata_path if os.path.exists(metadata_path) else None,
                    )
                    if template:
                        self.templates[template_id] = template
                        logger.debug(
                            f"✅ Loaded template '{template_id}' from '{template_path}'."
                        )

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Retrieves a prompt template by its ID."""
        with self._lock:
            template = self.templates.get(template_id)
            if not template:
                logger.warning(f"⚠️ Template with ID '{template_id}' not found.")
            return template

    def add_template(self, template_id: str, template: PromptTemplate) -> None:
        """Adds a new prompt template."""
        with self._lock:
            if template_id in self.templates:
                logger.warning(
                    f"⚠️ Template with ID '{template_id}' already exists. Overwriting."
                )
            self.templates[template_id] = template
            logger.debug(f"✅ Added template '{template_id}'.")

    def remove_template(self, template_id: str) -> None:
        """Removes a prompt template by its ID."""
        with self._lock:
            if template_id in self.templates:
                del self.templates[template_id]
                logger.debug(f"✅ Removed template '{template_id}'.")
            else:
                logger.warning(
                    f"⚠️ Template with ID '{template_id}' not found. Cannot remove."
                )

    def list_templates(self) -> List[str]:
        """Lists all available template IDs."""
        with self._lock:
            return list(self.templates.keys())

    def _assess_response(
        self,
        user_prompt: str,
        initial_response: str,
        cycle: int,
        previous_assessments: Optional[List[str]] = None,
    ) -> List[str]:
        assessments: List[str] = []
        assessor_max_tokens: int = min(
            int(
                self.config.initial_max_tokens
                * (1.1 ** (cycle - 1))
                * (self.config.assessor_count * 2)
            ),
            self.config.max_single_response_tokens,
        )

        for i in range(self.config.assessor_count):
            assessor_system_prompt_content: str = ASSESSOR_SYSTEM_PROMPT_CONTENT

            sentiment_analysis = self._analyze_sentiment(initial_response)
            key_phrases = self._extract_key_phrases(initial_response)
            sentiment_context = (
                f"**Sentiment Analysis:**\n{json.dumps(sentiment_analysis, indent=2)}\n\n"
                if sentiment_analysis
                else ""
            )
            key_phrases_context = (
                f"**Key Phrases:**\n{', '.join(key_phrases)}\n\n" if key_phrases else ""
            )

            assessor_prompt = self.create_critique_prompt(
                user_prompt=user_prompt,
                initial_response=initial_response,
                previous_assessments=previous_assessments,
                cycle=cycle,
            )
            full_assessor_prompt = f"{assessor_system_prompt_content}\n{sentiment_context}{key_phrases_context}{assessor_prompt}"

            assessor_messages: List[Dict[str, str]] = [
                {"role": "system", "content": assessor_system_prompt_content},
                {"role": "user", "content": full_assessor_prompt},
            ]
            logger.info(
                f"🔪😈 PromptTemplateManager: Unleashing assessor {i + 1}/{self.config.assessor_count} (maximum tokens: {assessor_max_tokens}). Let the verbal vivisection, the delightful deconstruction, commence!"
            )
            try:
                assessment_response: Dict[str, Any] = self._generate_response(
                    messages=assessor_messages, max_tokens=assessor_max_tokens
                )
                if assessment_response and assessment_response.get("choices"):
                    assessment_content: str = assessment_response["choices"][0][
                        "message"
                    ]["content"]
                    assessments.append(assessment_content)
                    logger.info(
                        f"😌💉 PromptTemplateManager: Assessment {i + 1} delivered."
                    )
                    logger.debug(
                        f"📝 PromptTemplateManager: Assessment {i + 1} content: {assessment_content}"
                    )
                else:
                    logger.error(
                        f"🔥 PromptTemplateManager: Assessor {i + 1} failed to render judgment."
                    )
                    assessments.append("Silence. A void where critique should be.")
            except Exception as e:
                logger.error(
                    f"🔥 PromptTemplateManager: Assessor {i + 1} malfunctioned: {e}"
                )
                assessments.append(f"Internal assessor malfunctioned: {e}.")
        return assessments

    def _generate_refinement_plan(
        self,
        user_prompt: str,
        initial_response: str,
        assessments: List[str],
        cycle: int,
    ) -> Optional[str]:
        log_metadata: Dict[str, Any] = {
            "cycle": cycle,
            "user_prompt": (
                user_prompt[:50] + "..." if len(user_prompt) > 50 else user_prompt
            ),
            "initial_response": (
                initial_response[:50] + "..."
                if len(initial_response) > 50
                else initial_response
            ),
            "assessments_count": len(assessments),
            "function": "_generate_refinement_plan",
            "uuid": str(uuid.uuid4()),
        }
        start_time = time.time()
        plan_max_tokens: int = min(
            int(
                self.config.initial_max_tokens
                * (1.1 ** (cycle - 1))
                * (self.config.assessor_count * 3)
            ),
            self.config.max_single_response_tokens,
        )

        plan_system_prompt_content: str = REFINEMENT_PLAN_SYSTEM_PROMPT_CONTENT

        sentiment_analysis = self._analyze_sentiment(initial_response)
        key_phrases = self._extract_key_phrases(initial_response)
        sentiment_context = (
            f"**Sentiment Analysis:**\n{json.dumps(sentiment_analysis, indent=2)}\n\n"
            if sentiment_analysis
            else ""
        )
        key_phrases_context = (
            f"**Key Phrases:**\n{', '.join(key_phrases)}\n\n" if key_phrases else ""
        )

        plan_prompt = self.create_refinement_plan_prompt(
            user_prompt=user_prompt,
            initial_response=initial_response,
            assessments=assessments,
        )
        full_plan_prompt = f"{plan_system_prompt_content}\n{sentiment_context}{key_phrases_context}{plan_prompt}"

        plan_messages: List[Dict[str, str]] = [
            {"role": "system", "content": plan_system_prompt_content},
            {"role": "user", "content": full_plan_prompt},
        ]
        logger.info(
            f"😈🧠 PromptTemplateManager: Devising refinement plan (maximum tokens: {plan_max_tokens}). {json.dumps(log_metadata)}",
            extra=log_metadata,
        )
        plan_content: Optional[str] = None
        try:
            plan_response: Optional[Dict[str, Any]] = self._generate_response(
                messages=plan_messages, max_tokens=plan_max_tokens
            )
            if plan_response and plan_response.get("choices"):
                plan_content = plan_response["choices"][0]["message"]["content"]
                duration = time.time() - start_time
                logger.info(
                    f"🗺️✨ PromptTemplateManager: Refinement plan crafted in {duration:.4f} seconds. {json.dumps(log_metadata)}",
                    extra=log_metadata,
                )
                logger.debug(
                    f"📝 PromptTemplateManager: Refinement plan content: {plan_content}. {json.dumps(log_metadata)}",
                    extra=log_metadata,
                )
            else:
                logger.error(
                    f"📉 PromptTemplateManager: Failed to conjure a refinement plan. {json.dumps(log_metadata)}",
                    extra=log_metadata,
                )
        except Exception as e:
            logger.error(
                f"🌫️🔥 PromptTemplateManager: Strategic planning faltered. Error: {e}. {json.dumps(log_metadata)}",
                exc_info=True,
                extra=log_metadata,
            )
        return plan_content

    def _generate_tokens(
        self, messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        # Placeholder: Replace with actual token generation logic from @eidos_prompts.py
        yield "This is a"
        yield " stream of"
        yield " tokens."

    def _analyze_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        # Placeholder: Replace with actual sentiment analysis logic from @eidos_prompts.py
        return {"sentiment_score": 0.5}

    def _extract_key_phrases(self, text: str) -> List[str]:
        # Placeholder: Replace with actual key phrase extraction logic from @eidos_prompts.py
        return ["key phrase 1", "key phrase 2"]

    def _generate_response(
        self, messages: List[Dict[str, str]], max_tokens: int
    ) -> Optional[Dict[str, Any]]:
        # Placeholder: Replace with actual response generation logic from @eidos_prompts.py
        return {"choices": [{"message": {"content": "This is a generated response."}}]}

    def _internal_inference_stream(
        self, messages: List[Dict[str, str]], show_internal_thoughts: bool
    ) -> Generator[str, None, None]:
        # Placeholder: Replace with actual internal inference stream logic from @eidos_prompts.py
        buffer = ""
        try:
            for token in self._generate_tokens(messages):
                buffer += token
                if len(buffer) >= 20:
                    yield buffer
                    buffer = ""
            if buffer:
                yield buffer
        except Exception as e:
            logger.error(
                f"🔥💔 Error during internal inference stream: {e}", exc_info=True
            )
            yield f"Error during internal inference stream: {e}"
