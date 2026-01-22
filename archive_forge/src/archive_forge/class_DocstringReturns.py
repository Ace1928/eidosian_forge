import enum
import typing as T
class DocstringReturns(DocstringMeta):
    """DocstringMeta symbolizing :returns or :yields metadata."""

    def __init__(self, args: T.List[str], description: T.Optional[str], type_name: T.Optional[str], is_generator: bool, return_name: T.Optional[str]=None) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.type_name = type_name
        self.is_generator = is_generator
        self.return_name = return_name