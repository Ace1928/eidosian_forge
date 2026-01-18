from __future__ import annotations
from typing import Callable, Iterable, Mapping, Pattern
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText
True when the word before the cursor matches.