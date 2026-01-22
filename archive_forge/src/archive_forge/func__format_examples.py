from dataclasses import dataclass, field
from typing import Any, Dict, List
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.prompt_template import PromptTemplate
def _format_examples(self):
    return '\n'.join(map(str, self.examples))