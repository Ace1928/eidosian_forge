import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def __list_dicts_callback(self, tag, name, desc, file):
    """Collector callback for listing dictionaries.

        This method is used as a callback into the _enchant function
        'enchant_broker_list_dicts'.  It collects the given arguments into
        an appropriate tuple and appends them to '__list_dicts_result'.
        """
    tag = tag.decode()
    name = name.decode()
    desc = desc.decode()
    file = file.decode()
    self.__list_dicts_result.append((tag, (name, desc, file)))