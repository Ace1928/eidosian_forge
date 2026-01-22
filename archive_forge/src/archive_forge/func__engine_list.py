from . import engines
from .exceptions import TemplateDoesNotExist
def _engine_list(using=None):
    return engines.all() if using is None else [engines[using]]