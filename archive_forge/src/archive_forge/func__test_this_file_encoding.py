import re
import fnmatch
def _test_this_file_encoding(fname, test_file, unicode_whitelist=unicode_whitelist, unicode_strict_whitelist=unicode_strict_whitelist):
    """Test helper function for unicode test

    The test may have to operate on filewise manner, so it had moved
    to a separate process.
    """
    has_unicode = False
    is_in_whitelist = False
    is_in_strict_whitelist = False
    for patt in unicode_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_whitelist = True
            break
    for patt in unicode_strict_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_strict_whitelist = True
            is_in_whitelist = True
            break
    if is_in_whitelist:
        for idx, line in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                has_unicode = True
        if not has_unicode and (not is_in_strict_whitelist):
            assert False, message_unicode_D % fname
    else:
        for idx, line in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                assert False, message_unicode_B % (fname, idx + 1)