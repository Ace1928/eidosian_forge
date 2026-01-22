from typing import Optional, Tuple
from catalogue import RegistryError
from wasabi import msg
from ..util import registry
from ._util import Arg, Opt, app

    Find the module, path and line number to the file the registered
    function is defined in, if available.

    func_name (str): Name of the registered function.
    registry_name (Optional[str]): Name of the catalogue registry.

    DOCS: https://spacy.io/api/cli#find-function
    