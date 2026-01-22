from __future__ import annotations
class EquivalentSiteSearchError(AbstractChemenvError):
    """Equivalent site search error."""

    def __init__(self, site):
        """
        Args:
            site:
        """
        self.site = site

    def __str__(self):
        return f'Equivalent site could not be found for the following site : {self.site}'