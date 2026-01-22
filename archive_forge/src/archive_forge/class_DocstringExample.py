import enum
import typing as T
class DocstringExample(DocstringMeta):
    """DocstringMeta symbolizing example metadata."""

    def __init__(self, args: T.List[str], snippet: T.Optional[str], description: T.Optional[str]) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.snippet = snippet
        self.description = description