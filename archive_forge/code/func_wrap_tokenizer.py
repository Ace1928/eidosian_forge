import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
def wrap_tokenizer(tk1, tk2):
    """Wrap one tokenizer inside another.

    This function takes two tokenizer functions 'tk1' and 'tk2',
    and returns a new tokenizer function that passes the output
    of tk1 through tk2 before yielding it to the calling code.
    """
    tkw = Filter(tk1)
    tkw._split = tk2
    return tkw