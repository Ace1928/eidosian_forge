from typing import Any, Optional
class UniverseMismatchError(ValueError):

    def __init__(self, client_universe, credentials_universe):
        message = f"The configured universe domain ({client_universe}) does not match the universe domain found in the credentials ({credentials_universe}). If you haven't configured the universe domain explicitly, `{DEFAULT_UNIVERSE}` is the default."
        super().__init__(message)