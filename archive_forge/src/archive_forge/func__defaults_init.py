from functools import partial
def _defaults_init():
    """
    create & return defaults for all reportlab settings from
    reportlab.rl_settings.py
    reportlab.local_rl_settings.py
    reportlab_settings.py or ~/.reportlab_settings

    latter values override earlier
    """
    from reportlab.lib.utils import rl_exec
    import os
    _DEFAULTS = {}
    rl_exec('from reportlab.rl_settings import *', _DEFAULTS)
    _overrides = _DEFAULTS.copy()
    try:
        rl_exec('from reportlab.local_rl_settings import *', _overrides)
        _DEFAULTS.update(_overrides)
    except ImportError:
        pass
    _overrides = _DEFAULTS.copy()
    try:
        rl_exec('from reportlab_settings import *', _overrides)
        _DEFAULTS.update(_overrides)
    except ImportError:
        _overrides = _DEFAULTS.copy()
        try:
            try:
                fn = os.path.expanduser(os.path.join('~', '.reportlab_settings'))
            except (KeyError, ImportError):
                fn = None
            if fn:
                with open(fn, 'rb') as f:
                    rl_exec(f.read(), _overrides)
                _DEFAULTS.update(_overrides)
        except:
            pass
    return _DEFAULTS