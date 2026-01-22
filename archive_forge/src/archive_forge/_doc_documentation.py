from typing import Any, Dict
A decorator: Generates docs for private methods/functions.

    For example:
    ```
    class Try:
      @doc_private
      def _private(self):
        ...
    ```
    As a rule of thumb, private (beginning with `_`) methods/functions are
    not documented. This decorator allows to force document a private
    method/function.

    Args:
      obj: The class-attribute to force the documentation for.
    Returns:
      obj
    