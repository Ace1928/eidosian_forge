from typing import Any, Optional
class VersionRequirementError(SphinxError):
    """Incompatible Sphinx version error."""
    category = 'Sphinx version error'