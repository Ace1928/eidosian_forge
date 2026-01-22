import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def __describe(self, check_this=True):
    """Return a tuple describing the dictionary.

        This method returns a four-element tuple describing the underlying
        spellchecker system providing the dictionary.  It will contain the
        following strings:

            * language tag
            * name of dictionary provider
            * description of dictionary provider
            * dictionary file

        Direct use of this method is not recommended - instead, access this
        information through the 'tag' and 'provider' attributes.
        """
    if check_this:
        self._check_this()
    _e.dict_describe(self._this, self.__describe_callback)
    return self.__describe_result