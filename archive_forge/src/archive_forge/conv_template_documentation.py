import os
import sys
import re
Find all named replacements in the header

    Returns a list of dictionaries, one for each loop iteration,
    where each key is a name to be substituted and the corresponding
    value is the replacement string.

    Also return a list of exclusions.  The exclusions are dictionaries
     of key value pairs. There can be more than one exclusion.
     [{'var1':'value1', 'var2', 'value2'[,...]}, ...]

    