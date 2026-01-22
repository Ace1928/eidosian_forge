import re
from typing import Optional, Pattern, Match, Optional
class LazyReCompile:
    """Compile regular expressions on first use

    This class allows one to store regular expressions and compiles them on
    first use."""

    def __init__(self, regex: str, flags: int=0) -> None:
        self.regex = regex
        self.flags = flags

    @cached_property
    def compiled(self) -> Pattern[str]:
        return re.compile(self.regex, self.flags)

    def finditer(self, *args, **kwargs):
        return self.compiled.finditer(*args, **kwargs)

    def search(self, *args, **kwargs) -> Optional[Match[str]]:
        return self.compiled.search(*args, **kwargs)

    def match(self, *args, **kwargs) -> Optional[Match[str]]:
        return self.compiled.match(*args, **kwargs)

    def sub(self, *args, **kwargs) -> str:
        return self.compiled.sub(*args, **kwargs)