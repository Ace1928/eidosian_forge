from . import exceptions
from . import misc
from . import normalizers
def allow_schemes(self, *schemes):
    """Require the scheme to be one of the provided schemes.

        .. versionadded:: 1.0

        :param schemes:
            Schemes, without ``://`` that are allowed.
        :returns:
            The validator instance.
        :rtype:
            Validator
        """
    for scheme in schemes:
        self.allowed_schemes.add(normalizers.normalize_scheme(scheme))
    return self