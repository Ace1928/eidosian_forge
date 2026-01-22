import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _print_locale():
    """ Test function.
    """
    categories = {}

    def _init_categories(categories=categories):
        for k, v in globals().items():
            if k[:3] == 'LC_':
                categories[k] = v
    _init_categories()
    del categories['LC_ALL']
    print('Locale defaults as determined by getdefaultlocale():')
    print('-' * 72)
    lang, enc = getdefaultlocale()
    print('Language: ', lang or '(undefined)')
    print('Encoding: ', enc or '(undefined)')
    print()
    print('Locale settings on startup:')
    print('-' * 72)
    for name, category in categories.items():
        print(name, '...')
        lang, enc = getlocale(category)
        print('   Language: ', lang or '(undefined)')
        print('   Encoding: ', enc or '(undefined)')
        print()
    print()
    print('Locale settings after calling resetlocale():')
    print('-' * 72)
    resetlocale()
    for name, category in categories.items():
        print(name, '...')
        lang, enc = getlocale(category)
        print('   Language: ', lang or '(undefined)')
        print('   Encoding: ', enc or '(undefined)')
        print()
    try:
        setlocale(LC_ALL, '')
    except:
        print('NOTE:')
        print('setlocale(LC_ALL, "") does not support the default locale')
        print('given in the OS environment variables.')
    else:
        print()
        print('Locale settings after calling setlocale(LC_ALL, ""):')
        print('-' * 72)
        for name, category in categories.items():
            print(name, '...')
            lang, enc = getlocale(category)
            print('   Language: ', lang or '(undefined)')
            print('   Encoding: ', enc or '(undefined)')
            print()