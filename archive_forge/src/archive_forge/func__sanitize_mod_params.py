import copy
import gettext
import locale
import logging
import os
import warnings
from oslo_i18n import _locale
from oslo_i18n import _translate
def _sanitize_mod_params(self, other):
    """Sanitize the object being modded with this Message.

        - Add support for modding 'None' so translation supports it
        - Trim the modded object, which can be a large dictionary, to only
        those keys that would actually be used in a translation
        - Snapshot the object being modded, in case the message is
        translated, it will be used as it was when the Message was created
        """
    if other is None:
        params = (other,)
    elif isinstance(other, dict):
        params = {}
        if isinstance(self.params, dict):
            params.update(((key, self._copy_param(val)) for key, val in self.params.items()))
        params.update(((key, self._copy_param(val)) for key, val in other.items()))
    else:
        params = self._copy_param(other)
    return params