import random
def RunFromIPython():
    if not correct_imports:
        return False
    try:
        return __IPYTHON__ is not None
    except NameError:
        return False