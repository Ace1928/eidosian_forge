import string
from typing import Any, List, Union
A prompt template for a language model.

    A prompt template consists of an array of strings that will be concatenated together. It accepts
    a set of parameters from the user that can be used to generate a prompt for a language model.

    The template can be formatted using f-strings.

    Example:

        .. code-block:: python

            from mlflow.metrics.genai.prompt_template import PromptTemplate

            # Instantiation using initializer
            prompt = PromptTemplate(template_str="Say {foo} {baz}")

            # Instantiation using partial_fill
            prompt = PromptTemplate(template_str="Say {foo} {baz}").partial_fill(foo="bar")

            # Format the prompt
            prompt.format(baz="qux")
    