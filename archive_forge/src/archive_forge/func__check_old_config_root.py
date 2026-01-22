import os
import platform
import pathlib
def _check_old_config_root():
    """
    Prior versions of keyring would search for the config
    in XDG_DATA_HOME, but should probably have been
    searching for config in XDG_CONFIG_HOME. If the
    config exists in the former but not in the latter,
    raise a RuntimeError to force the change.
    """
    globals()['_check_old_config_root'] = lambda: None
    config_file_new = os.path.join(_config_root_Linux(), 'keyringrc.cfg')
    config_file_old = os.path.join(_data_root_Linux(), 'keyringrc.cfg')
    if os.path.isfile(config_file_old) and (not os.path.isfile(config_file_new)):
        msg = f'Keyring config exists only in the old location {config_file_old} and should be moved to {config_file_new} to work with this version of keyring.'
        raise RuntimeError(msg)