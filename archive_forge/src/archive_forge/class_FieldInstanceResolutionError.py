from __future__ import annotations
import typing
class FieldInstanceResolutionError(MarshmallowError, TypeError):
    """Raised when schema to instantiate is neither a Schema class nor an instance."""