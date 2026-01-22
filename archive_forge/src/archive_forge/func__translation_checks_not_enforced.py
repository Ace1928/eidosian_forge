import re
from hacking import core
def _translation_checks_not_enforced(filename):
    return any((pat in filename for pat in ['/tests/', 'rally-jobs/plugins/']))