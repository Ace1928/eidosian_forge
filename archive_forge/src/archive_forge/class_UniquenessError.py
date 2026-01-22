from typing import Any, Optional
class UniquenessError(Error):
    """A uniqueness assumption was made in the context, and that is not true"""

    def __init__(self, values: Any):
        Error.__init__(self, 'Uniqueness assumption is not fulfilled. Multiple values are: %s' % values)