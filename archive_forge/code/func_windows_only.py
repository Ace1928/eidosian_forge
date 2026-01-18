import platform
def windows_only(func):
    if platform.system() != 'Windows':
        return lambda *args, **kwargs: None
    return func