import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def __describe_callback(self, tag, name, desc, file):
    """Collector callback for dictionary description.

        This method is used as a callback into the _enchant function
        'enchant_dict_describe'.  It collects the given arguments in
        a tuple and stores them in the attribute '__describe_result'.
        """
    tag = tag.decode()
    name = name.decode()
    desc = desc.decode()
    file = file.decode()
    self.__describe_result = (tag, name, desc, file)