import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like
def write_style_file(style, filename):
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False, compact=True)
    strl = pp.pformat(style).splitlines()
    if not isinstance(style, dict):
        raise TypeError('Specified style must be in `dict` format')
    if path.exists(filename):
        print('"' + filename + '" exists.')
        answer = input(' Overwrite(Y/N)? ')
        a = answer.lower()
        if a != 'y' and a != 'yes':
            raise FileExistsError
    f = open(filename, 'w')
    f.write('style = ' + strl[0].replace('{', 'dict(', 1).replace("'", '', 2).replace(':', ' =', 1) + '\n')
    for line in strl[1:-1]:
        if "'" in line[0:5]:
            f.write('            ' + line.replace("'", '', 2).replace(':', ' =', 1) + '\n')
        else:
            f.write('            ' + line + '\n')
    line = strl[-1]
    if "'" in line[0:5]:
        line = line.replace("'", '', 2).replace(':', ' =', 1)[::-1]
    else:
        line = line[::-1]
    f.write('            ' + line.replace('}', ')', 1)[::-1] + '\n')
    f.close()
    print('Wrote style file "' + filename + '"')
    return