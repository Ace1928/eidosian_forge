import functools
import re
import warnings
def _build_precedence_key(self, with_build=False):
    """Build a precedence key.

        The "build" component should only be used when sorting an iterable
        of versions.
        """
    if self.prerelease:
        prerelease_key = tuple((NumericIdentifier(part) if part.isdigit() else AlphaIdentifier(part) for part in self.prerelease))
    else:
        prerelease_key = (MaxIdentifier(),)
    if not with_build:
        return (self.major, self.minor, self.patch, prerelease_key)
    build_key = tuple((NumericIdentifier(part) if part.isdigit() else AlphaIdentifier(part) for part in self.build or ()))
    return (self.major, self.minor, self.patch, prerelease_key, build_key)