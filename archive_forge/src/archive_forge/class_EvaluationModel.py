from dataclasses import dataclass, field
from typing import Any, Dict, List
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.prompt_template import PromptTemplate
@dataclass
class EvaluationModel:
    """
    Useful to compute v1 prompt for make_genai_metric
    """
    name: str
    definition: str
    grading_prompt: str
    examples: List[EvaluationExample] = None
    model: str = default_model
    parameters: Dict[str, Any] = field(default_factory=lambda: default_parameters)

    def to_dict(self):
        examples_str = '' if self.examples is None or len(self.examples) == 0 else f'Examples:\n{self._format_examples()}'
        return {'model': self.model, 'eval_prompt': grading_system_prompt_template.partial_fill(name=self.name, definition=self.definition, grading_prompt=self.grading_prompt, examples=examples_str), 'parameters': self.parameters}

    def _format_examples(self):
        return '\n'.join(map(str, self.examples))